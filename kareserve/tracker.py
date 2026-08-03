# SPDX-License-Identifier: Apache-2.0
"""vLLM KV-event catalog and server-metric tracking."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import msgspec
import zmq
import zmq.asyncio

from kareserve.state import (
    CatalogStatus,
    MetricsStatus,
    NodeRoutingState,
    NodeState,
    PrefixMatch,
    RouteCandidate,
    SchedulerRequest,
)

logger = logging.getLogger("kareserve.tracker")
ExternalBlockHash = bytes | int
BlockIdentity = tuple[int, ExternalBlockHash]
END_SEQUENCE = -1


@dataclass(frozen=True, slots=True)
class GpuCachedBlock:
    parent_block_hash: ExternalBlockHash | None
    token_ids: tuple[int, ...]


class RawMapBlockStored(msgspec.Struct, tag="BlockStored", omit_defaults=True):
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


class RawMapBlockRemoved(msgspec.Struct, tag="BlockRemoved", omit_defaults=True):
    block_hashes: list[ExternalBlockHash]
    medium: str | None = None
    group_idx: int | None = None
    locality: str | None = None


class RawMapAllBlocksCleared(
    msgspec.Struct,
    tag="AllBlocksCleared",
    omit_defaults=True,
):
    pass


RawMapEvent = RawMapBlockStored | RawMapBlockRemoved | RawMapAllBlocksCleared


class RawMapEventBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float
    events: list[RawMapEvent]
    data_parallel_rank: int | None = None


class KareserveTracker:
    def __init__(self, initial_nodes: list[NodeState]) -> None:
        self.nodes = {node.node_id: node for node in initial_nodes}
        self.gpu_blocks: dict[str, dict[BlockIdentity, GpuCachedBlock]] = {
            node.node_id: {} for node in initial_nodes
        }
        self.gpu_block_index: dict[
            str,
            dict[
                tuple[int, ExternalBlockHash | None, tuple[int, ...]],
                BlockIdentity,
            ],
        ] = {node.node_id: {} for node in initial_nodes}
        self.gpu_block_sizes: dict[str, set[int]] = {
            node.node_id: set() for node in initial_nodes
        }
        self.gpu_catalog_status = {
            node.node_id: CatalogStatus.HEALTHY for node in initial_nodes
        }
        self.lmcache_catalog_status = {
            node.cache_domain_id: CatalogStatus.HEALTHY for node in initial_nodes
        }
        self.last_sequence: dict[str, int] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._sockets: list[zmq.asyncio.Socket] = []
        self._context = zmq.asyncio.Context.instance()
        self._decoder = msgspec.msgpack.Decoder(RawMapEventBatch)

    def get_routing_states(self) -> dict[str, NodeRoutingState]:
        routing_states: dict[str, NodeRoutingState] = {}
        for node_id, node in self.nodes.items():
            routing_states[node_id] = NodeRoutingState(
                node_id=node.node_id,
                host=node.host,
                port=node.port,
                cache_domain_id=node.cache_domain_id,
                router_active_requests=node.router_active_requests,
                router_inflight_work=node.router_inflight_work,
                running_requests=node.running_requests,
                waiting_requests=node.waiting_requests,
                kv_cache_usage=node.kv_cache_usage,
                gpu_total_blocks=node.gpu_total_blocks,
                gpu_block_size=node.gpu_block_size,
                metrics_status=node.metrics_status,
                metrics_updated_at=node.metrics_updated_at,
                process_start_time_seconds=node.process_start_time_seconds,
                gpu_catalog_status=self.gpu_catalog_status[node_id],
                lmcache_catalog_status=self.lmcache_catalog_status[
                    node.cache_domain_id
                ],
            )
        return routing_states

    def build_route_candidates(
        self,
        requests: Sequence[SchedulerRequest],
        default_block_size: int,
        external_matches: dict[str, dict[str, PrefixMatch]] | None = None,
    ) -> dict[str, dict[str, RouteCandidate]]:
        routing_states = self.get_routing_states()
        matrix: dict[str, dict[str, RouteCandidate]] = {}
        for request in requests:
            by_node: dict[str, RouteCandidate] = {}
            for node_id, node in routing_states.items():
                event_match = (
                    self._match_prefix(request.prompt_tokens, node)
                    if node.gpu_catalog_status is CatalogStatus.HEALTHY
                    else PrefixMatch(prompt_tokens=len(request.prompt_tokens))
                )
                external_match = (external_matches or {}).get(
                    request.request_id, {}
                ).get(node.cache_domain_id)
                match = PrefixMatch(
                    prompt_tokens=len(request.prompt_tokens),
                    gpu_prefix_tokens=event_match.gpu_prefix_tokens,
                    cpu_prefix_tokens=(
                        external_match.cpu_prefix_tokens if external_match else 0
                    ),
                    fs_prefix_tokens=(
                        external_match.fs_prefix_tokens if external_match else 0
                    ),
                    obj_prefix_tokens=(
                        external_match.obj_prefix_tokens if external_match else 0
                    ),
                )
                block_size = node.gpu_block_size or default_block_size
                required_tokens = (
                    max(0, len(request.prompt_tokens) - match.gpu_prefix_tokens)
                    + request.max_tokens
                )
                by_node[node_id] = RouteCandidate(
                    request_id=request.request_id,
                    node=node,
                    prefix_match=match,
                    required_new_gpu_blocks=(
                        ceil(required_tokens / block_size) if block_size > 0 else None
                    ),
                )
            matrix[request.request_id] = by_node
        return matrix

    def reserve_route(self, node_id: str, work: float) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.router_active_requests += 1
        node.router_inflight_work += max(0.0, work)

    def release_route(self, node_id: str, work: float) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.router_active_requests = max(0, node.router_active_requests - 1)
        node.router_inflight_work = max(0.0, node.router_inflight_work - max(0.0, work))

    def set_lmcache_lookup_status(self, failed_domains: set[str]) -> None:
        for domain_id in self.lmcache_catalog_status:
            self.lmcache_catalog_status[domain_id] = (
                CatalogStatus.DEGRADED
                if domain_id in failed_domains
                else CatalogStatus.HEALTHY
            )

    def update_metrics_text(self, node_id: str, metrics_text: str) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        process_start_samples = re.findall(
            r"^process_start_time_seconds\s+([-+0-9.eE]+)$",
            metrics_text,
            re.MULTILINE,
        )
        if process_start_samples:
            process_start = float(process_start_samples[-1])
            if (
                node.process_start_time_seconds is not None
                and process_start != node.process_start_time_seconds
            ):
                self._clear_gpu_catalog(node_id)
                self.last_sequence.pop(node_id, None)
            node.process_start_time_seconds = process_start
        metric_specs = {
            "running_requests": ("vllm:num_requests_running", sum),
            "waiting_requests": ("vllm:num_requests_waiting", sum),
            "kv_cache_usage": ("vllm:kv_cache_usage_perc", max),
            "lmcache_queries": (
                "vllm:external_prefix_cache_queries_total",
                sum,
            ),
            "lmcache_hits": (
                "vllm:external_prefix_cache_hits_total",
                sum,
            ),
        }
        for field, (metric, reducer) in metric_specs.items():
            matches = re.findall(
                rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
                metrics_text,
                re.MULTILINE,
            )
            if matches:
                value = reducer(float(item) for item in matches)
                if field in {"running_requests", "waiting_requests"}:
                    setattr(node, field, int(value))
                else:
                    setattr(node, field, value)

        config_samples = re.findall(
            r"^vllm:cache_config_info\{([^}]*)\}\s+[-+0-9.eE]+$",
            metrics_text,
            re.MULTILINE,
        )
        total_blocks = 0
        for labels in config_samples:
            match = re.search(r'num_gpu_blocks="(\d+)"', labels)
            if match:
                total_blocks += int(match.group(1))
            block_size = re.search(r'block_size="(\d+)"', labels)
            if block_size:
                node.gpu_block_size = int(block_size.group(1))
        if total_blocks > 0:
            node.gpu_total_blocks = total_blocks

        node.metrics_status = MetricsStatus.AVAILABLE
        node.metrics_updated_at = time.time()

    def mark_metrics_unavailable(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node is not None:
            node.metrics_status = MetricsStatus.UNAVAILABLE

    @staticmethod
    def _event_value(event: Any, name: str, default: Any = None) -> Any:
        if isinstance(event, dict):
            return event.get(name, default)
        return getattr(event, name, default)

    def _store_event(self, node: NodeState, event: Any) -> None:
        if str(self._event_value(event, "medium", "")).upper() != "GPU":
            return
        hashes = list(self._event_value(event, "block_hashes", []))
        token_ids = list(self._event_value(event, "token_ids", []))
        block_size = int(self._event_value(event, "block_size", 0))
        parent_hash = self._event_value(event, "parent_block_hash")
        group_idx = int(self._event_value(event, "group_idx", 0) or 0)
        if not hashes or block_size <= 0:
            return

        node.gpu_block_size = block_size
        self.gpu_block_sizes[node.node_id].add(block_size)
        blocks = self.gpu_blocks[node.node_id]
        block_index = self.gpu_block_index[node.node_id]
        cursor = 0
        current_parent = parent_hash
        for block_hash in hashes:
            block_tokens = tuple(token_ids[cursor : cursor + block_size])
            cursor += block_size
            if len(block_tokens) != block_size:
                self.gpu_catalog_status[node.node_id] = CatalogStatus.DEGRADED
                break
            identity = (group_idx, block_hash)
            block = blocks.get(identity)
            if block is None:
                block = GpuCachedBlock(
                    parent_block_hash=current_parent,
                    token_ids=block_tokens,
                )
                blocks[identity] = block
            block_index[(group_idx, current_parent, block_tokens)] = identity
            current_parent = block_hash

    def _remove_event(self, node: NodeState, event: Any) -> None:
        if str(self._event_value(event, "medium", "")).upper() != "GPU":
            return
        group_idx = int(self._event_value(event, "group_idx", 0) or 0)
        blocks = self.gpu_blocks[node.node_id]
        block_index = self.gpu_block_index[node.node_id]
        for block_hash in self._event_value(event, "block_hashes", []):
            identity = (group_idx, block_hash)
            block = blocks.pop(identity, None)
            if block is None:
                continue
            block_index.pop(
                (group_idx, block.parent_block_hash, block.token_ids), None
            )

    def _clear_gpu_catalog(self, node_id: str) -> None:
        self.gpu_blocks[node_id].clear()
        self.gpu_block_index[node_id].clear()
        self.gpu_block_sizes[node_id].clear()

    def process_raw_payload(self, node_id: str, payload_bytes: bytes) -> bool:
        try:
            batch = self._decoder.decode(payload_bytes)
        except msgspec.DecodeError as exc:
            logger.warning("Failed to decode vLLM KV event payload: %s", exc)
            self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
            return False

        node = self.nodes.get(node_id)
        if node is None:
            return False
        for event in batch.events:
            if isinstance(event, RawMapAllBlocksCleared):
                self._clear_gpu_catalog(node_id)
            elif isinstance(event, RawMapBlockRemoved):
                self._remove_event(node, event)
            else:
                self._store_event(node, event)
        return True

    def _longest_prefix(
        self,
        prompt_tokens: Sequence[int],
        node_id: str,
        group_idx: int = 0,
    ) -> int:
        longest = 0
        block_index = self.gpu_block_index[node_id]
        for block_size in self.gpu_block_sizes[node_id]:
            cursor = 0
            parent_hash: ExternalBlockHash | None = None
            while cursor + block_size <= len(prompt_tokens):
                block_tokens = tuple(prompt_tokens[cursor : cursor + block_size])
                identity = block_index.get(
                    (group_idx, parent_hash, block_tokens)
                )
                if identity is None:
                    break
                parent_hash = identity[1]
                cursor += block_size
            longest = max(longest, cursor)
        return longest

    def _match_prefix(
        self,
        prompt_tokens: Sequence[int],
        node: NodeRoutingState,
    ) -> PrefixMatch:
        return PrefixMatch(
            prompt_tokens=len(prompt_tokens),
            gpu_prefix_tokens=self._longest_prefix(prompt_tokens, node.node_id),
        )

    @property
    def gpu_cached_block_count(self) -> int:
        return sum(len(blocks) for blocks in self.gpu_blocks.values())

    async def _apply_sequence(
        self, node_id: str, sequence: int, payload: bytes
    ) -> None:
        previous = self.last_sequence.get(node_id)
        if previous is not None and sequence <= previous:
            return
        if self.process_raw_payload(node_id, payload):
            self.last_sequence[node_id] = sequence

    async def _request_replay(
        self,
        node_id: str,
        replay_endpoint: str,
        start_sequence: int,
    ) -> bool:
        socket = self._context.socket(zmq.DEALER)
        socket.connect(replay_endpoint)
        try:
            await socket.send_multipart((b"", start_sequence.to_bytes(8, "big")))
            first_sequence: int | None = None
            while True:
                frames = await asyncio.wait_for(socket.recv_multipart(), timeout=2.0)
                if frames and frames[0] == b"":
                    frames = frames[1:]
                if len(frames) == 3:
                    _, sequence_bytes, payload = frames
                elif len(frames) == 2:
                    sequence_bytes, payload = frames
                else:
                    return False
                sequence = int.from_bytes(sequence_bytes, "big", signed=True)
                if sequence == END_SEQUENCE or not payload:
                    break
                if first_sequence is None:
                    first_sequence = sequence
                await self._apply_sequence(node_id, sequence, payload)
            return first_sequence is None or first_sequence <= start_sequence
        except (TimeoutError, zmq.ZMQError):
            return False
        finally:
            socket.close(linger=0)

    async def _listen(
        self,
        node_id: str,
        endpoint: str,
        replay_endpoint: str | None,
    ) -> None:
        socket = self._context.socket(zmq.SUB)
        socket.connect(endpoint)
        socket.subscribe(b"")
        self._sockets.append(socket)
        if replay_endpoint:
            replay_ok = await self._request_replay(
                node_id, replay_endpoint, start_sequence=0
            )
            if not replay_ok:
                self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
        try:
            while True:
                parts = await socket.recv_multipart()
                if len(parts) != 3:
                    self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
                    continue
                _, sequence_bytes, payload = parts
                sequence = int.from_bytes(sequence_bytes, "big")
                previous = self.last_sequence.get(node_id)
                if previous is not None and sequence == 0 and previous > 0:
                    self._clear_gpu_catalog(node_id)
                    self.last_sequence.pop(node_id, None)
                    previous = None
                if previous is not None and sequence > previous + 1:
                    replay_ok = bool(replay_endpoint) and await self._request_replay(
                        node_id, replay_endpoint, previous + 1
                    )
                    if not replay_ok:
                        self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
                elif previous is None and sequence > 0:
                    self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
                await self._apply_sequence(node_id, sequence, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.gpu_catalog_status[node_id] = CatalogStatus.DEGRADED
            logger.exception("KV event listener failed for %s", node_id)
        finally:
            socket.close(linger=0)

    def start_zmq_listener(
        self,
        node_id: str,
        endpoint: str,
        replay_endpoint: str | None = None,
    ) -> None:
        self._tasks.append(
            asyncio.create_task(
                self._listen(node_id, endpoint, replay_endpoint),
                name=f"kv-events-{node_id}",
            )
        )

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._sockets.clear()
