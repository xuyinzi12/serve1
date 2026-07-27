# SPDX-License-Identifier: Apache-2.0
"""Routing policies for the Kareserve gateway."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SchedulerRequest:
    request_id: str
    prompt_tokens: List[int]
    prefix_hashes: List[str] = field(default_factory=list)
    max_tokens: int = 16
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedBlock:
    block_hash: bytes | int | str
    parent_block_hash: bytes | int | str | None
    token_ids: tuple[int, ...]
    full_prefix_tokens: tuple[int, ...]
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
    gpu_free_blocks: int = 1000
    cached_prefix_hashes: Set[str] = field(default_factory=set)
    cached_blocks: Dict[bytes | int | str, CachedBlock] = field(
        default_factory=dict
    )

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def observed_load(self) -> float:
        metrics_load = self.running_requests + self.waiting_requests
        return max(float(self.active_requests), float(metrics_load))

    def longest_cached_prefix_tokens(self, prompt_tokens: List[int]) -> int:
        if not prompt_tokens:
            return 0
        best = 0
        prompt = tuple(prompt_tokens)
        for block in self.cached_blocks.values():
            prefix = block.full_prefix_tokens
            if len(prefix) <= len(prompt) and prompt[: len(prefix)] == prefix:
                best = max(best, len(prefix))
        return best


class KareserveBasePolicy(ABC):
    name = "base"

    @abstractmethod
    def select_node(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        pass

    def select_batch(
        self,
        requests: List[SchedulerRequest],
        cluster_nodes: Dict[str, NodeState],
    ) -> Dict[str, NodeState]:
        """Assign a window while accounting for earlier assignments."""
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: Dict[str, NodeState] = {}
        ordered = sorted(
            requests,
            key=lambda req: max(
                (
                    node.longest_cached_prefix_tokens(req.prompt_tokens)
                    for node in cluster_nodes.values()
                ),
                default=0,
            ),
            reverse=True,
        )
        for request in ordered:
            node = self.select_node_with_load(request, cluster_nodes, virtual_load)
            if node is None:
                continue
            assignments[request.request_id] = node
            virtual_load[node.node_id] += self.request_work(request)
        return assignments

    def select_node_with_load(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
        virtual_load: Dict[str, float],
    ) -> Optional[NodeState]:
        return self.select_node(request, cluster_nodes)

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
    ) -> None:
        self.prefix_tokens_per_load_unit = prefix_tokens_per_load_unit
        self.queue_weight = queue_weight
        self.group_block_size = max(1, group_block_size)

    def _score(
        self,
        request: SchedulerRequest,
        node: NodeState,
        load: float,
    ) -> float:
        hit_tokens = node.longest_cached_prefix_tokens(request.prompt_tokens)
        affinity = hit_tokens / max(self.prefix_tokens_per_load_unit, 1.0)
        return affinity - self.queue_weight * load

    def select_node_with_load(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
        virtual_load: Dict[str, float],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        return max(
            cluster_nodes.values(),
            key=lambda node: self._score(
                request, node, virtual_load.get(node.node_id, node.observed_load)
            ),
        )

    def select_node(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        return self.select_node_with_load(
            request,
            cluster_nodes,
            {node_id: node.observed_load for node_id, node in cluster_nodes.items()},
        )

    def _shared_prefix_groups(
        self, requests: List[SchedulerRequest]
    ) -> List[List[SchedulerRequest]]:
        prefix_counts: Dict[tuple[int, ...], int] = {}
        request_prefixes: Dict[str, List[tuple[int, ...]]] = {}
        for request in requests:
            prefixes = []
            for end in range(
                self.group_block_size,
                len(request.prompt_tokens) + 1,
                self.group_block_size,
            ):
                prefix = tuple(request.prompt_tokens[:end])
                prefixes.append(prefix)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            request_prefixes[request.request_id] = prefixes

        groups: Dict[
            tuple[int, ...] | tuple[str, str], List[SchedulerRequest]
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
        requests: List[SchedulerRequest],
        cluster_nodes: Dict[str, NodeState],
    ) -> Dict[str, NodeState]:
        if not cluster_nodes:
            return {}
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: Dict[str, NodeState] = {}
        for group in self._shared_prefix_groups(requests):
            node = max(
                cluster_nodes.values(),
                key=lambda candidate: (
                    sum(
                        candidate.longest_cached_prefix_tokens(
                            request.prompt_tokens
                        )
                        for request in group
                    )
                    / max(self.prefix_tokens_per_load_unit, 1.0)
                    - self.queue_weight * virtual_load[candidate.node_id]
                ),
            )
            for request in group:
                assignments[request.request_id] = node
                virtual_load[node.node_id] += self.request_work(request)
        return assignments


class ExampleTemplatePolicy(KareserveBasePolicy):
    """Compatibility policy retained for existing experiments."""

    name = "example_template"

    def __init__(self, weight_prefix: float = 1.0, weight_queue: float = 1.0) -> None:
        self.weight_prefix = weight_prefix
        self.weight_queue = weight_queue

    def select_node(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        return max(
            cluster_nodes.values(),
            key=lambda node: (
                self.weight_prefix
                * (
                    node.longest_cached_prefix_tokens(request.prompt_tokens)
                    or sum(
                        1
                        for value in request.prefix_hashes
                        if value in node.cached_prefix_hashes
                    )
                )
                - self.weight_queue * node.observed_load
            ),
        )


class LeastLoadPolicy(KareserveBasePolicy):
    name = "least_load"

    def select_node(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        return min(cluster_nodes.values(), key=lambda node: node.observed_load)

    def select_node_with_load(
        self,
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
        virtual_load: Dict[str, float],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        return min(
            cluster_nodes.values(),
            key=lambda node: (
                virtual_load.get(node.node_id, node.observed_load),
                node.node_id,
            ),
        )

    def select_batch(
        self,
        requests: List[SchedulerRequest],
        cluster_nodes: Dict[str, NodeState],
    ) -> Dict[str, NodeState]:
        virtual_load = {
            node_id: node.observed_load for node_id, node in cluster_nodes.items()
        }
        assignments: Dict[str, NodeState] = {}
        for request in requests:
            node = self.select_node_with_load(request, cluster_nodes, virtual_load)
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
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        nodes = sorted(cluster_nodes.values(), key=lambda node: node.node_id)
        node = nodes[self._next_index % len(nodes)]
        self._next_index += 1
        return node

    def select_batch(
        self,
        requests: List[SchedulerRequest],
        cluster_nodes: Dict[str, NodeState],
    ) -> Dict[str, NodeState]:
        assignments: Dict[str, NodeState] = {}
        for request in requests:
            node = self.select_node(request, cluster_nodes)
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
        request: SchedulerRequest,
        cluster_nodes: Dict[str, NodeState],
    ) -> Optional[NodeState]:
        if not cluster_nodes:
            return None
        nodes = sorted(cluster_nodes.values(), key=lambda node: node.node_id)
        prefix = request.prompt_tokens[: self.prefix_hash_tokens]
        key = ",".join(str(token) for token in prefix).encode("ascii")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        index = int.from_bytes(digest, byteorder="big") % len(nodes)
        return nodes[index]


class KareserveMediumAwarePolicy(WindowedPrefixAffinityPolicy):
    """Compatibility wrapper for the original hardware-profile constructor."""

    name = "medium_aware"

    def __init__(self, hardware_profile: Optional[Dict[str, Any]] = None) -> None:
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
