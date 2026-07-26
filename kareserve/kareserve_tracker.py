# SPDX-License-Identifier: Apache-2.0
"""vLLM KV-event and server-metric tracking."""

import copy
import json
import logging
import re
import threading
from typing import Any, Dict, List

from kareserve.kareserve_policy import CachedBlock, NodeState

try:
    import msgspec
    import zmq

    HAS_VLLM_EVENT_DEPS = True
except ImportError:
    msgspec = None
    zmq = None
    HAS_VLLM_EVENT_DEPS = False

logger = logging.getLogger("kareserve.tracker")
WireBlockHash = bytes | int


if HAS_VLLM_EVENT_DEPS:

    class RawBlockStored(
        msgspec.Struct,
        array_like=True,
        tag="BlockStored",
        omit_defaults=True,
    ):
        block_hashes: List[WireBlockHash]
        parent_block_hash: WireBlockHash | None
        token_ids: List[int]
        block_size: int
        lora_id: int | None = None
        medium: str | None = None
        lora_name: str | None = None
        extra_keys: List[Any] | None = None
        group_idx: int | None = None
        kv_cache_spec_kind: str | None = None
        kv_cache_spec_sliding_window: int | None = None
        locality: str | None = None

    class RawBlockRemoved(
        msgspec.Struct,
        array_like=True,
        tag="BlockRemoved",
        omit_defaults=True,
    ):
        block_hashes: List[WireBlockHash]
        medium: str | None
        group_idx: int | None = None
        locality: str | None = None

    class RawAllBlocksCleared(
        msgspec.Struct,
        array_like=True,
        tag="AllBlocksCleared",
        omit_defaults=True,
    ):
        pass

    RawEvent = RawBlockStored | RawBlockRemoved | RawAllBlocksCleared

    class RawEventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
        ts: float
        events: List[RawEvent]
        data_parallel_rank: int | None = None


class KareserveTracker:
    def __init__(self, initial_nodes: List[NodeState]) -> None:
        self.nodes = {node.node_id: node for node in initial_nodes}
        self._lock = threading.Lock()
        self._running = False
        self._threads: List[threading.Thread] = []
        self._decoder = (
            msgspec.msgpack.Decoder(RawEventBatch)
            if HAS_VLLM_EVENT_DEPS
            else None
        )

    def get_node_states(self) -> Dict[str, NodeState]:
        with self._lock:
            return copy.deepcopy(self.nodes)

    def update_active_requests(self, node_id: str, delta: int) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is not None:
                node.active_requests = max(0, node.active_requests + delta)

    def update_metrics_text(self, node_id: str, metrics_text: str) -> None:
        names = {
            "running_requests": "vllm:num_requests_running",
            "waiting_requests": "vllm:num_requests_waiting",
            "kv_cache_usage": "vllm:kv_cache_usage_perc",
        }
        values: Dict[str, float] = {}
        for field, metric in names.items():
            matches = re.findall(
                rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
                metrics_text,
                re.MULTILINE,
            )
            if matches:
                values[field] = sum(float(value) for value in matches)
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return
            for field, value in values.items():
                setattr(node, field, value)

    @staticmethod
    def _event_value(event: Any, name: str, default: Any = None) -> Any:
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)

    def _store_event(self, node: NodeState, event: Any) -> None:
        hashes = list(self._event_value(event, "block_hashes", []))
        token_ids = list(self._event_value(event, "token_ids", []))
        block_size = int(self._event_value(event, "block_size", 0))
        parent_hash = self._event_value(event, "parent_block_hash")
        medium = self._event_value(event, "medium", None) or "GPU"
        if not hashes or block_size <= 0:
            return

        cursor = 0
        current_parent = parent_hash
        for block_hash in hashes:
            block_tokens = tuple(token_ids[cursor : cursor + block_size])
            cursor += block_size
            parent = node.cached_blocks.get(current_parent)
            parent_tokens = parent.full_prefix_tokens if parent is not None else ()
            full_prefix = parent_tokens + block_tokens
            node.cached_blocks[block_hash] = CachedBlock(
                block_hash=block_hash,
                parent_block_hash=current_parent,
                token_ids=block_tokens,
                full_prefix_tokens=full_prefix,
                medium=medium,
            )
            node.cached_prefix_hashes.add(str(block_hash))
            current_parent = block_hash

    def _remove_event(self, node: NodeState, event: Any) -> None:
        for block_hash in self._event_value(event, "block_hashes", []):
            node.cached_blocks.pop(block_hash, None)
            node.cached_prefix_hashes.discard(str(block_hash))

    def process_raw_payload(self, node_id: str, payload_bytes: bytes) -> None:
        try:
            if self._decoder is not None:
                events = self._decoder.decode(payload_bytes).events
            else:
                events = json.loads(payload_bytes.decode("utf-8")).get("events", [])
            with self._lock:
                node = self.nodes.get(node_id)
                if node is None:
                    return
                for event in events:
                    event_name = type(event).__name__
                    if isinstance(event, dict):
                        event_name = event.get("type", "")
                    if event_name in {"RawAllBlocksCleared", "AllBlocksCleared"}:
                        node.cached_blocks.clear()
                        node.cached_prefix_hashes.clear()
                    elif event_name in {"RawBlockRemoved", "BlockRemoved"} or (
                        isinstance(event, dict)
                        and "medium" in event
                        and "token_ids" not in event
                    ):
                        self._remove_event(node, event)
                    else:
                        self._store_event(node, event)
        except Exception as exc:
            logger.warning("Failed to decode vLLM KV event payload: %s", exc)

    def start_zmq_listener(self, node_id: str, zmq_endpoint: str) -> None:
        if not HAS_VLLM_EVENT_DEPS:
            raise RuntimeError("msgspec and pyzmq are required for KV event tracking")

        def listener_loop() -> None:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.connect(zmq_endpoint)
            sock.subscribe(b"")
            while self._running:
                try:
                    parts = sock.recv_multipart()
                    if len(parts) == 3:
                        self.process_raw_payload(node_id, parts[2])
                except zmq.Again:
                    continue
                except Exception as exc:
                    logger.error("KV event listener failed for %s: %s", node_id, exc)
            sock.close(linger=0)

        self._running = True
        thread = threading.Thread(
            target=listener_loop, daemon=True, name=f"kv-events-{node_id}"
        )
        self._threads.append(thread)
        thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
