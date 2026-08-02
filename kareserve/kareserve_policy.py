# SPDX-License-Identifier: Apache-2.0
"""Routing policies for the Kareserve gateway."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kareserve.kareserve_state import (
    CacheMedium,
    MetricsStatus,
    RouteAssignment,
    RouteCandidate,
    SchedulerRequest,
)

CandidateMatrix = dict[str, dict[str, RouteCandidate]]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Convert cache loading and model work into one cost unit."""

    tokens_per_work_unit: float = 256.0
    decode_token_weight: float = 4.0
    cpu_load_weight: float = 0.25
    fs_load_weight: float = 2.0
    obj_load_weight: float = 4.0
    unknown_load_weight: float = 1.0
    compute_ms_per_token: float | None = None
    kv_bytes_per_token: float | None = None
    medium_profiles: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_hardware_profile(
        cls,
        profile: dict[str, Any],
        *,
        tokens_per_work_unit: float,
        decode_token_weight: float,
    ) -> CostModel:
        gpu_flops = float(profile.get("gpu_flops_tflops", 0.0)) * 1e12
        model_params = float(profile.get("model_params_billions", 0.0)) * 1e9
        layers = int(profile.get("num_layers", 0))
        hidden_size = int(profile.get("hidden_size", 0))
        dtype_bytes = float(profile.get("kv_dtype_bytes", 2.0))
        compute_ms = None
        kv_bytes = None
        if gpu_flops > 0 and model_params > 0:
            compute_ms = 2.0 * model_params / gpu_flops * 1000.0
        if layers > 0 and hidden_size > 0:
            kv_bytes = 2.0 * layers * hidden_size * dtype_bytes

        medium_profiles = dict(profile.get("medium_profiles", {}))
        h2d_bandwidth = float(profile.get("h2d_bandwidth_gbps", 0.0))
        if h2d_bandwidth > 0 and "CPU" not in medium_profiles:
            medium_profiles["CPU"] = {
                "bandwidth_gbps": h2d_bandwidth,
                "base_latency_ms": float(profile.get("h2d_base_latency_ms", 0.0)),
            }

        return cls(
            tokens_per_work_unit=max(tokens_per_work_unit, 1.0),
            decode_token_weight=max(decode_token_weight, 0.0),
            cpu_load_weight=float(profile.get("cpu_load_weight", 0.25)),
            fs_load_weight=float(profile.get("fs_load_weight", 2.0)),
            obj_load_weight=float(profile.get("obj_load_weight", 4.0)),
            unknown_load_weight=float(profile.get("unknown_load_weight", 1.0)),
            compute_ms_per_token=compute_ms,
            kv_bytes_per_token=kv_bytes,
            medium_profiles=medium_profiles,
        )

    def compute_cost(self, tokens: int) -> float:
        if self.compute_ms_per_token is not None:
            return max(0, tokens) * self.compute_ms_per_token
        return max(0, tokens) / self.tokens_per_work_unit

    def load_cost(self, medium: CacheMedium, tokens: int) -> float:
        tokens = max(0, tokens)
        profile = (self.medium_profiles or {}).get(medium.value)
        if profile and self.kv_bytes_per_token is not None:
            bandwidth = float(profile.get("bandwidth_gbps", 0.0))
            if bandwidth > 0:
                transfer_ms = (
                    tokens * self.kv_bytes_per_token / (bandwidth * 1e9) * 1000.0
                )
                return transfer_ms + float(profile.get("base_latency_ms", 0.0))
        weights = {
            CacheMedium.CPU: self.cpu_load_weight,
            CacheMedium.FS: self.fs_load_weight,
            CacheMedium.OBJ: self.obj_load_weight,
            CacheMedium.UNKNOWN: self.unknown_load_weight,
        }
        return tokens * weights.get(medium, 0.0) / self.tokens_per_work_unit

    def candidate_work(
        self, request: SchedulerRequest, candidate: RouteCandidate
    ) -> float:
        match = candidate.prefix_match
        external_options: list[tuple[CacheMedium, int]] = []
        for medium, prefix_tokens in (
            (CacheMedium.CPU, match.cpu_prefix_tokens),
            (CacheMedium.FS, match.fs_prefix_tokens),
            (CacheMedium.OBJ, match.obj_prefix_tokens),
            (CacheMedium.UNKNOWN, match.unknown_prefix_tokens),
        ):
            if prefix_tokens > match.gpu_prefix_tokens:
                external_options.append((medium, prefix_tokens))

        if external_options:
            longest_external = max(item[1] for item in external_options)
            options = []
            for medium, prefix_tokens in external_options:
                if prefix_tokens != longest_external:
                    continue
                loaded_tokens = prefix_tokens - match.gpu_prefix_tokens
                missing_tokens = max(0, match.prompt_tokens - prefix_tokens)
                options.append(
                    self.load_cost(medium, loaded_tokens)
                    + self.compute_cost(missing_tokens)
                )
            prompt_cost = min(options)
        else:
            prompt_cost = self.compute_cost(
                match.prompt_tokens - match.gpu_prefix_tokens
            )
        decode_cost = self.compute_cost(
            int(self.decode_token_weight * request.max_tokens)
        )
        return prompt_cost + decode_cost


class KareserveBasePolicy(ABC):
    name = "base"

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    @abstractmethod
    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        raise NotImplementedError

    @staticmethod
    def _eligible(
        candidates: Iterable[RouteCandidate],
    ) -> list[RouteCandidate]:
        values = list(candidates)
        available = [
            item
            for item in values
            if item.node.metrics_status is MetricsStatus.AVAILABLE
        ]
        return available or values

    def _assignment(
        self,
        request: SchedulerRequest,
        candidate: RouteCandidate,
        estimated_cost: float,
    ) -> RouteAssignment:
        work = self.cost_model.candidate_work(request, candidate)
        return RouteAssignment(
            candidate=candidate,
            inflight_work=work,
            estimated_cost=estimated_cost,
        )


class WindowedPrefixAffinityPolicy(KareserveBasePolicy):
    """Joint prefix, load, transfer, and capacity routing."""

    name = "windowed_prefix"

    def __init__(
        self,
        *,
        cost_model: CostModel,
        queue_weight: float = 1.0,
        group_block_size: int = 16,
        kv_cache_weight: float = 2.0,
        kv_cache_high_watermark: float = 0.80,
        kv_cache_hard_limit: float = 0.95,
    ) -> None:
        super().__init__(cost_model)
        self.queue_weight = max(queue_weight, 0.0)
        self.group_block_size = max(1, group_block_size)
        self.kv_cache_weight = max(0.0, kv_cache_weight)
        self.kv_cache_high_watermark = min(max(kv_cache_high_watermark, 0.0), 1.0)
        self.kv_cache_hard_limit = min(
            max(kv_cache_hard_limit, self.kv_cache_high_watermark), 1.0
        )

    def _capacity_pressure(self, candidate: RouteCandidate) -> float:
        node = candidate.node
        usage = node.kv_cache_usage
        if usage is None or usage <= self.kv_cache_high_watermark:
            return 0.0
        remaining = max(1.0 - self.kv_cache_high_watermark, 1e-6)
        normalized = (min(usage, 1.0) - self.kv_cache_high_watermark) / remaining
        pressure = self.kv_cache_weight * normalized * normalized
        free_blocks = node.estimated_gpu_free_blocks
        required_blocks = candidate.required_new_gpu_blocks
        if (
            free_blocks is not None
            and required_blocks is not None
            and required_blocks > free_blocks
        ):
            pressure += self.kv_cache_weight * (
                (required_blocks - free_blocks) / max(required_blocks, 1)
            )
        if usage >= self.kv_cache_hard_limit:
            pressure += self.kv_cache_weight
        return pressure

    def _base_load(self, candidate: RouteCandidate, virtual_work: float) -> float:
        request_pressure = 0.25 * candidate.node.router_active_requests
        return virtual_work + self.queue_weight * (
            candidate.node.queue_depth + request_pressure
        )

    def _candidate_cost(
        self,
        request: SchedulerRequest,
        candidate: RouteCandidate,
        virtual_work: float,
    ) -> float:
        return (
            self._base_load(candidate, virtual_work)
            + self.cost_model.candidate_work(request, candidate)
            + self._capacity_pressure(candidate)
        )

    def _shared_prefix_groups(
        self, requests: list[SchedulerRequest]
    ) -> list[list[SchedulerRequest]]:
        prefix_counts: dict[tuple[int, ...], int] = {}
        request_prefixes: dict[str, list[tuple[int, ...]]] = {}
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

        grouped: dict[object, list[SchedulerRequest]] = {}
        for request in requests:
            shared = [
                prefix
                for prefix in request_prefixes[request.request_id]
                if prefix_counts[prefix] > 1
            ]
            key: object = (
                max(shared, key=len) if shared else ("request", request.request_id)
            )
            grouped.setdefault(key, []).append(request)
        return sorted(grouped.values(), key=len, reverse=True)

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        virtual_work: dict[str, float] = {}
        for by_node in candidates.values():
            for candidate in by_node.values():
                virtual_work.setdefault(
                    candidate.node.node_id,
                    candidate.node.router_inflight_work,
                )

        assignments: dict[str, RouteAssignment] = {}
        for group in self._shared_prefix_groups(requests):
            common_nodes = set(candidates.get(group[0].request_id, {}))
            for request in group[1:]:
                common_nodes &= set(candidates.get(request.request_id, {}))
            if not common_nodes:
                continue

            representative = candidates[group[0].request_id]
            eligible = self._eligible(
                representative[node_id] for node_id in common_nodes
            )
            node_id = min(
                (item.node.node_id for item in eligible),
                key=lambda candidate_node_id: sum(
                    self._candidate_cost(
                        request,
                        candidates[request.request_id][candidate_node_id],
                        virtual_work[candidate_node_id],
                    )
                    for request in group
                ),
            )
            for request in group:
                candidate = candidates[request.request_id][node_id]
                cost = self._candidate_cost(request, candidate, virtual_work[node_id])
                assignment = self._assignment(request, candidate, cost)
                assignments[request.request_id] = assignment
                virtual_work[node_id] += assignment.inflight_work
        return assignments


class LeastLoadPolicy(KareserveBasePolicy):
    name = "least_load"

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        virtual_work: dict[str, float] = {}
        assignments: dict[str, RouteAssignment] = {}
        for request in requests:
            eligible = self._eligible(candidates.get(request.request_id, {}).values())
            if not eligible:
                continue
            for candidate in eligible:
                virtual_work.setdefault(
                    candidate.node.node_id,
                    candidate.node.router_inflight_work,
                )
            candidate = min(
                eligible,
                key=lambda item: (
                    virtual_work[item.node.node_id] + item.node.queue_depth,
                    item.node.node_id,
                ),
            )
            cost = virtual_work[candidate.node.node_id] + candidate.node.queue_depth
            assignment = self._assignment(request, candidate, cost)
            assignments[request.request_id] = assignment
            virtual_work[candidate.node.node_id] += assignment.inflight_work
        return assignments


class RoundRobinPolicy(KareserveBasePolicy):
    name = "round_robin"

    def __init__(self, cost_model: CostModel | None = None) -> None:
        super().__init__(cost_model)
        self._next_index = 0

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        assignments: dict[str, RouteAssignment] = {}
        for request in requests:
            eligible = sorted(
                self._eligible(candidates.get(request.request_id, {}).values()),
                key=lambda item: item.node.node_id,
            )
            if not eligible:
                continue
            candidate = eligible[self._next_index % len(eligible)]
            self._next_index += 1
            assignments[request.request_id] = self._assignment(request, candidate, 0.0)
        return assignments


class PrefixHashPolicy(KareserveBasePolicy):
    name = "prefix_hash"

    def __init__(
        self,
        prefix_hash_tokens: int = 256,
        cost_model: CostModel | None = None,
    ) -> None:
        super().__init__(cost_model)
        self.prefix_hash_tokens = max(1, prefix_hash_tokens)

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        assignments: dict[str, RouteAssignment] = {}
        for request in requests:
            eligible = sorted(
                self._eligible(candidates.get(request.request_id, {}).values()),
                key=lambda item: item.node.node_id,
            )
            if not eligible:
                continue
            prefix = request.prompt_tokens[: self.prefix_hash_tokens]
            key = ",".join(str(token) for token in prefix).encode("ascii")
            digest = hashlib.blake2b(key, digest_size=8).digest()
            candidate = eligible[int.from_bytes(digest, "big") % len(eligible)]
            assignments[request.request_id] = self._assignment(request, candidate, 0.0)
        return assignments


class KareserveMediumAwarePolicy(WindowedPrefixAffinityPolicy):
    """Compatibility name for the medium-aware windowed policy."""

    name = "medium_aware"
