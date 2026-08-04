# SPDX-License-Identifier: Apache-2.0
"""Routing policies for the Kareserve gateway."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kareserve.performance import PolynomialPrefillModel
from kareserve.state import (
    CacheMedium,
    MetricsStatus,
    PrefixMatch,
    RouteAssignment,
    RouteCandidate,
    RouteCostBreakdown,
    SchedulerRequest,
)

CandidateMatrix = dict[str, dict[str, RouteCandidate]]


@dataclass(frozen=True, slots=True)
class PrefixGroup:
    requests: tuple[SchedulerRequest, ...]
    shared_prefix_tokens: int


@dataclass(slots=True)
class _AssignmentPlan:
    score: float
    assignments: dict[str, RouteAssignment]
    virtual_work: dict[str, float]
    virtual_free_blocks: dict[str, int | None]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Convert cache loading and model work into one cost unit."""

    tokens_per_work_unit: float = 256.0
    decode_token_weight: float = 4.0
    cpu_load_weight: float = 0.25
    fs_load_weight: float = 2.0
    obj_load_weight: float = 4.0
    compute_ms_per_token: float | None = None
    kv_bytes_per_token: float | None = None
    medium_profiles: dict[str, dict[str, float]] | None = None
    prefill_model: PolynomialPrefillModel | None = None

    @classmethod
    def from_hardware_profile(
        cls,
        profile: dict[str, Any],
        *,
        tokens_per_work_unit: float,
        decode_token_weight: float,
    ) -> CostModel:
        layers = int(profile.get("num_layers", 0))
        hidden_size = int(profile.get("hidden_size", 0))
        dtype_bytes = float(profile.get("kv_dtype_bytes", 2.0))
        configured_prefill_ms = profile.get("prefill_ms_per_token")
        compute_ms = (
            float(configured_prefill_ms)
            if configured_prefill_ms is not None
            and float(configured_prefill_ms) > 0
            else None
        )
        kv_bytes = None
        if layers > 0 and hidden_size > 0:
            kv_bytes = 2.0 * layers * hidden_size * dtype_bytes

        medium_profiles = dict(profile.get("medium_profiles", {}))
        memory_to_gpu_bandwidth = float(
            profile.get("host_memory_to_gpu_bandwidth_gbps", 0.0)
        )
        if memory_to_gpu_bandwidth > 0 and "CPU" not in medium_profiles:
            medium_profiles["CPU"] = {
                "bandwidth_gbps": memory_to_gpu_bandwidth,
                "base_latency_ms": float(
                    profile.get("host_memory_to_gpu_latency_ms", 0.0)
                ),
            }

        return cls(
            tokens_per_work_unit=max(tokens_per_work_unit, 1.0),
            decode_token_weight=max(decode_token_weight, 0.0),
            cpu_load_weight=float(profile.get("cpu_load_weight", 0.25)),
            fs_load_weight=float(profile.get("fs_load_weight", 2.0)),
            obj_load_weight=float(profile.get("obj_load_weight", 4.0)),
            compute_ms_per_token=compute_ms,
            kv_bytes_per_token=kv_bytes,
            medium_profiles=medium_profiles,
            prefill_model=PolynomialPrefillModel.from_profile(profile),
        )

    def compute_cost(self, tokens: int) -> float:
        if self.compute_ms_per_token is not None:
            return max(0, tokens) * self.compute_ms_per_token
        return max(0, tokens) / self.tokens_per_work_unit

    def prefill_cost(self, prompt_tokens: int, cached_prefix_tokens: int) -> float:
        if self.prefill_model is not None:
            return self.prefill_model.predict_ms(prompt_tokens, cached_prefix_tokens)
        return self.compute_cost(prompt_tokens - cached_prefix_tokens)

    def transfer_cost(self, medium: CacheMedium, tokens: int) -> float:
        tokens = max(0, tokens)
        profile = (self.medium_profiles or {}).get(medium.value)
        if (
            profile
            and self.kv_bytes_per_token is not None
        ):
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
        }
        return tokens * weights.get(medium, 0.0) / self.tokens_per_work_unit

    def prefix_cost(
        self, match: PrefixMatch, prompt_tokens: int | None = None
    ) -> float:
        total_tokens = match.prompt_tokens if prompt_tokens is None else prompt_tokens
        gpu_prefix = min(match.gpu_prefix_tokens, total_tokens)
        cpu_prefix = min(match.cpu_prefix_tokens, total_tokens)
        fs_prefix = min(match.fs_prefix_tokens, total_tokens)
        obj_prefix = min(match.obj_prefix_tokens, total_tokens)
        external_prefix = max(cpu_prefix, fs_prefix, obj_prefix)
        if external_prefix > gpu_prefix:
            gpu_load_tokens = external_prefix - gpu_prefix
            prompt_cost = self.transfer_cost(
                CacheMedium.CPU, gpu_load_tokens
            )
            l1_prefix = max(gpu_prefix, cpu_prefix)
            if external_prefix > l1_prefix:
                l2_options = [
                    self.transfer_cost(medium, external_prefix - l1_prefix)
                    for medium, prefix_tokens in (
                        (CacheMedium.FS, fs_prefix),
                        (CacheMedium.OBJ, obj_prefix),
                    )
                    if prefix_tokens == external_prefix
                ]
                prompt_cost += min(l2_options)
            prompt_cost += self.prefill_cost(total_tokens, external_prefix)
        else:
            prompt_cost = self.prefill_cost(total_tokens, gpu_prefix)
        return prompt_cost

    def candidate_prefill_cost(self, candidate: RouteCandidate) -> float:
        return self.prefix_cost(candidate.prefix_match)

    def candidate_work(
        self, request: SchedulerRequest, candidate: RouteCandidate
    ) -> float:
        prompt_cost = self.candidate_prefill_cost(candidate)
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
        cost_breakdown: RouteCostBreakdown | None = None,
    ) -> RouteAssignment:
        work = self.cost_model.candidate_work(request, candidate)
        return RouteAssignment(
            candidate=candidate,
            inflight_work=work,
            estimated_cost=estimated_cost,
            cost_breakdown=cost_breakdown,
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
        inflight_prefix_reuse_probability: float = 0.0,
        inflight_work_to_queue_time_scale: float = 1.0,
    ) -> None:
        super().__init__(cost_model)
        self.queue_weight = max(queue_weight, 0.0)
        self.group_block_size = max(1, group_block_size)
        self.kv_cache_weight = max(0.0, kv_cache_weight)
        self.kv_cache_high_watermark = min(max(kv_cache_high_watermark, 0.0), 1.0)
        self.kv_cache_hard_limit = min(
            max(kv_cache_hard_limit, self.kv_cache_high_watermark), 1.0
        )
        self.inflight_prefix_reuse_probability = min(
            max(inflight_prefix_reuse_probability, 0.0), 1.0
        )
        self.inflight_work_to_queue_time_scale = max(
            inflight_work_to_queue_time_scale, 0.0
        )

    def _capacity_pressure(
        self, candidate: RouteCandidate, free_blocks: int | None
    ) -> float:
        node = candidate.node
        usage = (
            1.0 - free_blocks / node.gpu_total_blocks
            if free_blocks is not None
            and node.gpu_total_blocks is not None
            and node.gpu_total_blocks > 0
            else node.kv_cache_usage
        )
        if usage is None or usage <= self.kv_cache_high_watermark:
            return 0.0
        remaining = max(1.0 - self.kv_cache_high_watermark, 1e-6)
        normalized = (min(usage, 1.0) - self.kv_cache_high_watermark) / remaining
        pressure = self.kv_cache_weight * normalized * normalized
        if usage >= self.kv_cache_hard_limit:
            pressure += self.kv_cache_weight
        required = candidate.required_new_gpu_blocks
        if (
            free_blocks is not None
            and required is not None
            and required > free_blocks
        ):
            pressure += self.kv_cache_weight * (
                1.0 + (required - free_blocks) / max(required, 1)
            )
        return pressure

    def _cost_breakdown(
        self,
        candidate: RouteCandidate,
        virtual_work: float,
        free_blocks: int | None,
    ) -> RouteCostBreakdown:
        request_pressure = 0.25 * candidate.node.router_active_requests
        return RouteCostBreakdown(
            prefill_cost=self.cost_model.candidate_prefill_cost(candidate),
            router_load_cost=(
                virtual_work * self.inflight_work_to_queue_time_scale
            ),
            engine_queue_cost=self.queue_weight
            * (candidate.node.queue_depth + request_pressure),
            capacity_cost=self._capacity_pressure(candidate, free_blocks),
        )

    def _candidate_cost(
        self,
        candidate: RouteCandidate,
        virtual_work: float,
        free_blocks: int | None,
    ) -> float:
        return self._cost_breakdown(candidate, virtual_work, free_blocks).total

    def _shared_prefix_groups(
        self, requests: list[SchedulerRequest]
    ) -> list[PrefixGroup]:
        deepest_group: dict[str, tuple[int, object]] = {}
        active_groups: list[list[SchedulerRequest]] = [requests]
        block_start = 0
        while active_groups:
            next_groups: list[list[SchedulerRequest]] = []
            for active_group in active_groups:
                buckets: dict[tuple[int, ...], list[SchedulerRequest]] = {}
                for request in active_group:
                    block_end = block_start + self.group_block_size
                    if len(request.prompt_tokens) < block_end:
                        continue
                    block = tuple(request.prompt_tokens[block_start:block_end])
                    buckets.setdefault(block, []).append(request)
                for bucket in buckets.values():
                    if len(bucket) < 2:
                        continue
                    marker = object()
                    for request in bucket:
                        deepest_group[request.request_id] = (block_end, marker)
                    next_groups.append(bucket)
            active_groups = next_groups
            block_start += self.group_block_size

        grouped: dict[object, list[SchedulerRequest]] = {}
        shared_lengths: dict[object, int] = {}
        for request in requests:
            shared = deepest_group.get(request.request_id)
            key: object = (
                shared[1] if shared else ("request", request.request_id)
            )
            grouped.setdefault(key, []).append(request)
            shared_lengths[key] = shared[0] if shared else 0
        values = [
            PrefixGroup(tuple(group), shared_lengths[key])
            for key, group in grouped.items()
        ]
        return sorted(
            values,
            key=lambda group: (
                group.shared_prefix_tokens,
                len(group.requests),
            ),
            reverse=True,
        )

    @staticmethod
    def _fits_capacity(
        candidate: RouteCandidate, free_blocks: int | None
    ) -> bool:
        required = candidate.required_new_gpu_blocks
        return required is None or free_blocks is None or required <= free_blocks

    def _build_plan(
        self,
        group: PrefixGroup,
        candidates: CandidateMatrix,
        virtual_work: dict[str, float],
        virtual_free_blocks: dict[str, int | None],
        forced_node_id: str | None = None,
        allow_capacity_fallback: bool = False,
    ) -> _AssignmentPlan | None:
        work = dict(virtual_work)
        free = dict(virtual_free_blocks)
        assignments: dict[str, RouteAssignment] = {}
        score = 0.0

        def largest_request(request: SchedulerRequest) -> int:
            values = candidates.get(request.request_id, {}).values()
            return max(
                (candidate.required_new_gpu_blocks or 0 for candidate in values),
                default=0,
            )

        ordered_requests = sorted(
            group.requests,
            key=largest_request,
            reverse=True,
        )
        for request in ordered_requests:
            eligible = self._eligible(
                candidates.get(request.request_id, {}).values()
            )
            if forced_node_id is not None:
                eligible = [
                    candidate
                    for candidate in eligible
                    if candidate.node.node_id == forced_node_id
                ]
            if not eligible:
                return None
            capacity_eligible = [
                candidate
                for candidate in eligible
                if self._fits_capacity(
                    candidate, free.get(candidate.node.node_id)
                )
            ]
            if capacity_eligible:
                eligible = capacity_eligible
            elif not allow_capacity_fallback:
                return None
            candidate = min(
                eligible,
                key=lambda item: (
                    self._candidate_cost(
                        item,
                        work[item.node.node_id],
                        free.get(item.node.node_id),
                    ),
                    item.node.node_id,
                ),
            )
            node_id = candidate.node.node_id
            breakdown = self._cost_breakdown(
                candidate, work[node_id], free.get(node_id)
            )
            cost = breakdown.total
            assignment = self._assignment(
                request, candidate, cost, cost_breakdown=breakdown
            )
            assignments[request.request_id] = assignment
            score += cost
            work[node_id] += assignment.inflight_work
            required = candidate.required_new_gpu_blocks
            if free.get(node_id) is not None and required is not None:
                free[node_id] = max(0, free[node_id] - required)

        if forced_node_id is not None and group.shared_prefix_tokens > 0:
            shared_costs = [
                self.cost_model.prefix_cost(
                    assignments[request.request_id].candidate.prefix_match,
                    group.shared_prefix_tokens,
                )
                for request in group.requests
            ]
            avoidable_cost = sum(shared_costs) - min(shared_costs)
            score -= self.inflight_prefix_reuse_probability * avoidable_cost
        return _AssignmentPlan(score, assignments, work, free)

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        virtual_work: dict[str, float] = {}
        virtual_free_blocks: dict[str, int | None] = {}
        for by_node in candidates.values():
            for candidate in by_node.values():
                virtual_work.setdefault(
                    candidate.node.node_id,
                    candidate.node.router_inflight_work,
                )
                virtual_free_blocks.setdefault(
                    candidate.node.node_id,
                    candidate.node.estimated_gpu_free_blocks,
                )

        assignments: dict[str, RouteAssignment] = {}
        for group in self._shared_prefix_groups(requests):
            independent = self._build_plan(
                group,
                candidates,
                virtual_work,
                virtual_free_blocks,
                allow_capacity_fallback=True,
            )
            best_plan = independent
            common_nodes = set(candidates.get(group.requests[0].request_id, {}))
            for request in group.requests[1:]:
                common_nodes &= set(candidates.get(request.request_id, {}))
            if len(group.requests) > 1:
                for node_id in sorted(common_nodes):
                    colocated = self._build_plan(
                        group,
                        candidates,
                        virtual_work,
                        virtual_free_blocks,
                        forced_node_id=node_id,
                    )
                    if colocated is not None and (
                        best_plan is None or colocated.score < best_plan.score
                    ):
                        best_plan = colocated
            if best_plan is None:
                continue
            assignments.update(best_plan.assignments)
            virtual_work = best_plan.virtual_work
            virtual_free_blocks = best_plan.virtual_free_blocks
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
