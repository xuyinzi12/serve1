# SPDX-License-Identifier: Apache-2.0
"""vLLM KV-event and server-metric tracking."""

import logging
import re
import threading
import time
from _thread import LockType
from typing import Any, Sequence

from kareserve.kareserve_policy import CachedBlock, NodeState, NodeRoutingState

import msgspec
import zmq

logger = logging.getLogger("kareserve.tracker")
ExternalBlockHash = bytes | int


class RawBlockStored(
    msgspec.Struct,
    array_like=True,
    tag="BlockStored",
    omit_defaults=True,
):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = None
    lora_name: str | None = None
    extra_keys: list[Any] | None = None
    group_idx: int | None = None #需要确认 vLLM 发出的不同组是否会在当前索引中发生混淆
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None

class RawBlockRemoved(
    msgspec.Struct,
    array_like=True,
    tag="BlockRemoved",
    omit_defaults=True,
):
    block_hashes: list[ExternalBlockHash]
    medium: str | None = None #如果只删除 GPU 副本，外部缓存副本可能仍然存在。但当前 Tracker 的删除实现没有读取 medium
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
    events: list[RawEvent]
    data_parallel_rank: int | None = None

class RawMapBlockStored(
    msgspec.Struct,
    tag="BlockStored",
    omit_defaults=True,
):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = None
    lora_name: str | None = None
    extra_keys: list[Any] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None

class RawMapBlockRemoved(
    msgspec.Struct,
    tag="BlockRemoved",
    omit_defaults=True,
):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None

class RawMapAllBlocksCleared(
    msgspec.Struct,
    tag="AllBlocksCleared",
    omit_defaults=True,
):
    pass

RawMapEvent = (
    RawMapBlockStored | RawMapBlockRemoved | RawMapAllBlocksCleared
)

class RawMapEventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float
    events: list[RawMapEvent]
    data_parallel_rank: int | None = None

class KareserveTracker:
    def __init__(self, initial_nodes: list[NodeState]) -> None:
        self.nodes = {node.node_id: node for node in initial_nodes}
        self._nodes_lock = threading.Lock()
        self._node_locks = {node.node_id: threading.Lock() for node in initial_nodes}
        self._running = False
        self._threads: list[threading.Thread] = []
        self._decoders = (
            msgspec.msgpack.Decoder(RawMapEventBatch),
            msgspec.msgpack.Decoder(RawEventBatch),
        )
        self._decoder_index_by_node: dict[str, int] = {} #用于记录每个节点上一次成功使用的解码器编号

    def _get_node_and_lock(
        self, node_id: str
    ) -> tuple[NodeState | None, LockType | None]:
        with self._nodes_lock:
            return self.nodes.get(node_id), self._node_locks.get(node_id)

    def get_routing_states(
        self, batch_prompt_tokens: Sequence[Sequence[int]] #表示这一轮可能同时为一批请求计算路由状态。
    ) -> dict[str, NodeRoutingState]:
        scores: dict[str, NodeRoutingState] = {}
        with self._nodes_lock:
            node_entries = [
                (node_id, node, self._node_locks.get(node_id))
                for node_id, node in self.nodes.items()
            ] #三元组：节点 ID NodeState 对象 该节点对应的锁

        for node_id, node, lock in node_entries:
            if lock is None:
                continue
            with lock:
                matched_blocks = []
                for prompt_tokens in batch_prompt_tokens:
                    matched = 0
                    node_block_size = node.block_size
                    if node_block_size > 0:
                        current_parent = None
                        cursor = 0
                        while cursor + node_block_size <= len(prompt_tokens):
                            block_tokens = tuple(prompt_tokens[cursor : cursor + node_block_size])
                            cursor += node_block_size
                            block_hash = node.block_index.get((current_parent, block_tokens))
                            if block_hash is None:
                                break
                            current_parent = block_hash
                            matched += 1
                    matched_blocks.append(matched)

                scores[node_id] = NodeRoutingState(
                    node_id=node_id,
                    host=node.host,
                    port=node.port,
                    matched_prefix_blocks=tuple(matched_blocks),
                    active_requests=node.active_requests,
                    running_requests=node.running_requests,
                    waiting_requests=node.waiting_requests,
                    kv_cache_usage=node.kv_cache_usage,
                    external_cache_queries=node.external_cache_queries,
                    external_cache_hits=node.external_cache_hits,
                    metrics_available=node.metrics_available,
                    metrics_updated_at=node.metrics_updated_at,
                    gpu_free_blocks=node.gpu_free_blocks,
                    block_size=node.block_size,
                )
        return scores

    def update_active_requests(self, node_id: str, delta: int) -> None:
        node, lock = self._get_node_and_lock(node_id)
        if node is not None and lock is not None:
            with lock:
                node.active_requests = max(0, node.active_requests + delta)

    def update_metrics_text(self, node_id: str, metrics_text: str) -> None:
        names = {
            "running_requests": ("vllm:num_requests_running", sum),
            "waiting_requests": ("vllm:num_requests_waiting", sum),
            "kv_cache_usage": ("vllm:kv_cache_usage_perc", max),
            "external_cache_queries": (
                "vllm:external_prefix_cache_queries_total",
                sum,
            ),
            "external_cache_hits": (
                "vllm:external_prefix_cache_hits_total",
                sum,
            ),
        }
        values: dict[str, float] = {}
        for field, (metric, reducer) in names.items():
            matches = re.findall(
                rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
                metrics_text,
                re.MULTILINE,
            )
            if matches:
                values[field] = reducer(float(value) for value in matches)
        with self._nodes_lock:
            node = self.nodes.get(node_id)
            lock = self._node_locks.get(node_id)
        if node is None or lock is None:
            return
        with lock:
            for field, value in values.items():
                setattr(node, field, value)
            node.metrics_available = True
            node.metrics_updated_at = time.time()

    def mark_metrics_unavailable(self, node_id: str) -> None:
        node, lock = self._get_node_and_lock(node_id)
        if node is not None and lock is not None:
            with lock:
                node.metrics_available = False

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

        node.block_size = block_size
        cursor = 0
        current_parent = parent_hash
        for block_hash in hashes:
            block_tokens = tuple(token_ids[cursor : cursor + block_size])
            cursor += block_size
            node.cached_blocks[block_hash] = CachedBlock(
                block_hash=block_hash,
                parent_block_hash=current_parent,
                token_ids=block_tokens,
                medium=medium,
            )
            node.block_index[(current_parent, block_tokens)] = block_hash
            node.cached_prefix_hashes.add(str(block_hash))
            current_parent = block_hash

    def _remove_event(self, node: NodeState, event: Any) -> None:
        for block_hash in self._event_value(event, "block_hashes", []):
            block = node.cached_blocks.pop(block_hash, None)
            if block is not None:
                node.block_index.pop((block.parent_block_hash, block.token_ids), None)
            node.cached_prefix_hashes.discard(str(block_hash))

    def _decode_batch(self, node_id: str, payload_bytes: bytes) -> Any:
        preferred = self._decoder_index_by_node.get(node_id)
        indexes = (preferred, 1 - preferred) if preferred is not None else (0, 1)
        last_error: msgspec.DecodeError | None = None
        for index in indexes:
            try:
                batch = self._decoders[index].decode(payload_bytes)
                self._decoder_index_by_node[node_id] = index
                return batch
            except msgspec.DecodeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def process_raw_payload(self, node_id: str, payload_bytes: bytes) -> None:
        try:
            batch = self._decode_batch(node_id, payload_bytes)
        except msgspec.DecodeError as exc:
            logger.warning("Failed to decode vLLM KV event payload: %s", exc)
            return

        node, lock = self._get_node_and_lock(node_id)
        if node is None or lock is None:
            return

        with lock:
            for event in batch.events:
                event_name = type(event).__name__
                if isinstance(event, dict):
                    event_name = event.get("type", "")
                if event_name in {
                    "RawAllBlocksCleared",
                    "RawMapAllBlocksCleared",
                    "AllBlocksCleared",
                }:
                    node.cached_blocks.clear()
                    node.block_index.clear()
                    node.cached_prefix_hashes.clear()
                elif event_name in {
                    "RawBlockRemoved",
                    "RawMapBlockRemoved",
                    "BlockRemoved",
                } or (
                    isinstance(event, dict)
                    and "medium" in event
                    and "token_ids" not in event
                ):
                    self._remove_event(node, event)
                else:
                    self._store_event(node, event)

    def start_zmq_listener(self, node_id: str, zmq_endpoint: str) -> None:
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
