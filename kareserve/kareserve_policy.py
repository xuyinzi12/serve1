# SPDX-License-Identifier: Apache-2.0
"""Routing policies for the Kareserve gateway."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SchedulerRequest:
    request_id: str
    prompt_tokens: list[int]
    prefix_hashes: list[str] = field(default_factory=list)
    max_tokens: int = 16
    raw_body: dict[str, Any] = field(default_factory=dict)

@dataclass
class CachedBlock:
    block_hash: bytes | int | str
    parent_block_hash: bytes | int | str | None
    token_ids: tuple[int, ...]
    medium: str = "GPU"

@dataclass
class NodeState:
    """Observable state for one independently addressable vLLM server."""

    node_id: str
    host: str
    port: int
    active_requests: int = 0
    running_requests: int = 0
    waiting_requests: int = 0
    kv_cache_usage: float = 0.0
    external_cache_queries: float = 0.0
    external_cache_hits: float = 0.0
    metrics_available: bool = False
    metrics_updated_at: float = 0.0
    gpu_free_blocks: int = 1000
    cached_prefix_hashes: set[str] = field(default_factory=set)
    cached_blocks: dict[bytes | int | str, CachedBlock] = field(
        default_factory=dict
    )
    block_index: dict[tuple[bytes | int | str | None, tuple[int, ...]], bytes | int | str] = field(
        default_factory=dict
    )
    block_size: int = 0

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

@dataclass(frozen=True, slots=True)
class NodeRoutingState:
    node_id: str
    host: str
    port: int
    matched_prefix_blocks: tuple[int, ...]
    active_requests: int
    running_requests: float
    waiting_requests: float
    kv_cache_usage: float
    external_cache_queries: float
    external_cache_hits: float
    metrics_available: bool
    metrics_updated_at: float
    gpu_free_blocks: int
    block_size: int

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def observed_load(self) -> float:
        metrics_load = self.running_requests + self.waiting_requests
        return max(float(self.active_requests), float(metrics_load))

    @property
    def external_cache_hit_rate(self) -> float:
        if self.external_cache_queries <= 0:
            return 0.0
        return self.external_cache_hits / self.external_cache_queries

class KareserveBasePolicy(ABC):
    name = "base"

    @abstractmethod
    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        pass

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> dict[str, NodeRoutingState]:
        """Assign a window while accounting for earlier assignments."""
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: dict[str, NodeRoutingState] = {}
        ordered = sorted(
            range(len(requests)),
            key=lambda req_idx: max(
                (
                    node.matched_prefix_blocks[req_idx]
                    for node in cluster_nodes.values()
                ),
                default=0,
            ),
            reverse=True,
        )
        for req_idx in ordered:
            request = requests[req_idx]
            node = self.select_node_with_load(req_idx, request, cluster_nodes, virtual_load)
            if node is None:
                continue
            assignments[request.request_id] = node
            virtual_load[node.node_id] += self.request_work(request)
        return assignments

    def select_node_with_load(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
        virtual_load: dict[str, float],
    ) -> NodeRoutingState | None:
        return self.select_node(req_idx, request, cluster_nodes)

    @staticmethod
    def available_nodes(cluster_nodes: dict[str, NodeRoutingState]) -> list[NodeRoutingState]:
        nodes = list(cluster_nodes.values())
        available = [node for node in nodes if node.metrics_available]
        return available or nodes

    @staticmethod
    def request_work(request: SchedulerRequest) -> float:
        return max(1.0, len(request.prompt_tokens) / 256.0)

class WindowedPrefixAffinityPolicy(KareserveBasePolicy):
    """Greedy prefix-affinity policy with per-window virtual load."""

    name = "windowed_prefix"

    def __init__(
        self,
        prefix_tokens_per_load_unit: float = 256.0,
        queue_weight: float = 1.0,
        group_block_size: int = 16,
        kv_cache_weight: float = 2.0,
        kv_cache_high_watermark: float = 0.80,
        kv_cache_hard_limit: float = 0.95,
        decode_token_weight: float = 4.0,
    ) -> None:
        self.prefix_tokens_per_load_unit = prefix_tokens_per_load_unit
        self.queue_weight = queue_weight
        self.group_block_size = max(1, group_block_size)
        self.kv_cache_weight = max(0.0, kv_cache_weight)
        self.kv_cache_high_watermark = min(
            max(kv_cache_high_watermark, 0.0), 1.0
        )
        self.kv_cache_hard_limit = min(
            max(kv_cache_hard_limit, self.kv_cache_high_watermark), 1.0
        )
        self.decode_token_weight = max(0.0, decode_token_weight)

    def _candidate_nodes(
        self, cluster_nodes: dict[str, NodeRoutingState]
    ) -> list[NodeRoutingState]:
        nodes = self.available_nodes(cluster_nodes)
        below_limit = [
            node for node in nodes if node.kv_cache_usage < self.kv_cache_hard_limit
        ]
        if below_limit:
            return below_limit
        if not nodes:
            return []
        minimum_usage = min(node.kv_cache_usage for node in nodes)
        return [
            node for node in nodes if node.kv_cache_usage == minimum_usage
        ]

    def _capacity_pressure(self, node: NodeRoutingState) -> float:
        usage = min(max(node.kv_cache_usage, 0.0), 1.0)
        if usage <= self.kv_cache_high_watermark:
            return 0.0
        remaining = max(1.0 - self.kv_cache_high_watermark, 1e-6)
        normalized = (usage - self.kv_cache_high_watermark) / remaining
        return self.kv_cache_weight * normalized * normalized

    def _request_work_for_node(
        self, req_idx: int, request: SchedulerRequest, node: NodeRoutingState
    ) -> float:
        hit_blocks = node.matched_prefix_blocks[req_idx]
        uncached_prompt_tokens = max(0, len(request.prompt_tokens) - hit_blocks * node.block_size)
        weighted_tokens = (
            uncached_prompt_tokens
            + self.decode_token_weight * max(0, request.max_tokens)
        )
        return max(
            1.0,
            weighted_tokens / max(self.prefix_tokens_per_load_unit, 1.0),
        )

    def _score(
        self,
        req_idx: int,
        request: SchedulerRequest,
        node: NodeRoutingState,
        load: float,
    ) -> float:
        hit_blocks = node.matched_prefix_blocks[req_idx]
        affinity = (
            hit_blocks * node.block_size
            / max(self.prefix_tokens_per_load_unit, 1.0)
        )
        projected_load = load + 0.5 * self._request_work_for_node(req_idx, request, node)
        return (
            affinity
            - self.queue_weight * projected_load
            - self._capacity_pressure(node)
        )

    def select_node_with_load(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
        virtual_load: dict[str, float],
    ) -> NodeRoutingState | None:
        candidates = self._candidate_nodes(cluster_nodes)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda node: self._score(
                req_idx, request, node, virtual_load.get(node.node_id, node.observed_load)
            ),
        )

    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        return self.select_node_with_load(
            req_idx,
            request,
            cluster_nodes,
            {node_id: node.observed_load for node_id, node in cluster_nodes.items()},
        )

    def _shared_prefix_groups(
        self, requests: list[SchedulerRequest], block_size: int = 16
    ) -> list[list[SchedulerRequest]]:
        prefix_counts: dict[tuple[int, ...], int] = {}
        request_prefixes: dict[str, list[tuple[int, ...]]] = {}
        for request in requests:
            prefixes = []
            for end in range(
                block_size,
                len(request.prompt_tokens) + 1,
                block_size,
            ):
                prefix = tuple(request.prompt_tokens[:end])
                prefixes.append(prefix)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            request_prefixes[request.request_id] = prefixes

        groups: dict[
            tuple[int, ...] | tuple[str, str], list[SchedulerRequest]
        ] = {}
        for request in requests:
            shared = [
                prefix
                for prefix in request_prefixes[request.request_id]
                if prefix_counts[prefix] > 1
            ]
            key: tuple[int, ...] | tuple[str, str]
            key = max(shared, key=len) if shared else ("request", request.request_id)
            groups.setdefault(key, []).append(request)
        return sorted(
            groups.values(),
            key=lambda group: (
                len(group),
                max((len(request.prompt_tokens) for request in group), default=0),
            ),
            reverse=True,
        )

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> dict[str, NodeRoutingState]:
        candidates = self._candidate_nodes(cluster_nodes)
        if not candidates:
            return {}
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: dict[str, NodeRoutingState] = {}

        # Build mapping from request_id to request index for O(1) lookup
        req_idx_map = {req.request_id: i for i, req in enumerate(requests)}

        # Assuming homogeneous block size across the cluster for grouping
        block_size = next(
            (node.block_size for node in candidates if node.block_size > 0),
            self.group_block_size,
        )

        for group in self._shared_prefix_groups(requests, block_size):
            node = max(
                candidates,
                key=lambda candidate: (
                    sum(
                        candidate.matched_prefix_blocks[req_idx_map[request.request_id]]
                        for request in group
                    )
                    / max(self.prefix_tokens_per_load_unit, 1.0)
                    - self.queue_weight
                    * (
                        virtual_load[candidate.node_id]
                        + 0.5
                        * sum(
                            self._request_work_for_node(req_idx_map[request.request_id], request, candidate)
                            for request in group
                        )
                    )
                    - self._capacity_pressure(candidate)
                ),
            )
            for request in group:
                assignments[request.request_id] = node
                virtual_load[node.node_id] += self._request_work_for_node(
                    req_idx_map[request.request_id], request, node
                )
        return assignments

class ExampleTemplatePolicy(KareserveBasePolicy):
    """Compatibility policy retained for existing experiments."""

    name = "example_template"

    def __init__(self, weight_prefix: float = 1.0, weight_queue: float = 1.0) -> None:
        self.weight_prefix = weight_prefix
        self.weight_queue = weight_queue

    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        nodes = self.available_nodes(cluster_nodes)
        if not nodes:
            return None
        return max(
            nodes,
            key=lambda node: (
                self.weight_prefix
                * node.matched_prefix_blocks[req_idx]
                - self.weight_queue * node.observed_load
            ),
        )

class LeastLoadPolicy(KareserveBasePolicy):
    name = "least_load"

    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        nodes = self.available_nodes(cluster_nodes)
        if not nodes:
            return None
        return min(nodes, key=lambda node: node.observed_load)

    def select_node_with_load(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
        virtual_load: dict[str, float],
    ) -> NodeRoutingState | None:
        nodes = self.available_nodes(cluster_nodes)
        if not nodes:
            return None
        return min(
            nodes,
            key=lambda node: (
                virtual_load.get(node.node_id, node.observed_load),
                node.node_id,
            ),
        )

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> dict[str, NodeRoutingState]:
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: dict[str, NodeRoutingState] = {}
        for req_idx, request in enumerate(requests):
            node = self.select_node_with_load(req_idx, request, cluster_nodes, virtual_load)
            if node is None:
                continue
            assignments[request.request_id] = node
            virtual_load[node.node_id] += self.request_work(request)
        return assignments

class RoundRobinPolicy(KareserveBasePolicy):
    """Deterministic request-order round robin baseline."""

    name = "round_robin"

    def __init__(self) -> None:
        self._next_index = 0

    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        nodes = sorted(
            self.available_nodes(cluster_nodes), key=lambda node: node.node_id
        )
        if not nodes:
            return None
        node = nodes[self._next_index % len(nodes)]
        self._next_index += 1
        return node

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> dict[str, NodeRoutingState]:
        assignments: dict[str, NodeRoutingState] = {}
        for req_idx, request in enumerate(requests):
            node = self.select_node(req_idx, request, cluster_nodes)
            if node is not None:
                assignments[request.request_id] = node
        return assignments

class PrefixHashPolicy(KareserveBasePolicy):
    """Stable prefix-to-node mapping baseline."""

    name = "prefix_hash"

    def __init__(self, prefix_hash_tokens: int = 256) -> None:
        self.prefix_hash_tokens = max(1, prefix_hash_tokens)

    def select_node(
        self,
        req_idx: int,
        request: SchedulerRequest,
        cluster_nodes: dict[str, NodeRoutingState],
    ) -> NodeRoutingState | None:
        nodes = sorted(
            self.available_nodes(cluster_nodes), key=lambda node: node.node_id
        )
        if not nodes:
            return None
        prefix = request.prompt_tokens[: self.prefix_hash_tokens]
        key = ",".join(str(token) for token in prefix).encode("ascii")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        index = int.from_bytes(digest, byteorder="big") % len(nodes)
        return nodes[index]

class KareserveMediumAwarePolicy(WindowedPrefixAffinityPolicy):
    """Compatibility wrapper for the original hardware-profile constructor."""

    name = "medium_aware"

    def __init__(self, hardware_profile: dict[str, Any] | None = None) -> None:
        profile = hardware_profile or {}
        self.gpu_flops = profile.get("gpu_flops_tflops", 312.0) * 1e12
        self.model_params = profile.get("model_params_billions", 7.0) * 1e9
        self.num_layers = profile.get("num_layers", 32)
        self.hidden_size = profile.get("hidden_size", 4096)
        self.medium_profiles = profile.get(
            "medium_profiles",
            {
                "GPU": {"bandwidth_gbps": 2000.0, "base_latency_ms": 0.01},
                "CPU": {"bandwidth_gbps": 32.0, "base_latency_ms": 0.5},
            },
        )
        super().__init__()

    def estimate_compute_latency_ms(self, prompt_tokens_len: int) -> float:
        return (
            2.0 * prompt_tokens_len * self.model_params / self.gpu_flops
        ) * 1000.0

    def estimate_load_latency_ms(self, prompt_tokens_len: int, medium: str) -> float:
        profile = self.medium_profiles.get(medium, self.medium_profiles["CPU"])
        size = 2 * self.num_layers * self.hidden_size * 2 * prompt_tokens_len
        return (
            size / (profile["bandwidth_gbps"] * 1e9) * 1000.0
            + profile["base_latency_ms"]
        )
