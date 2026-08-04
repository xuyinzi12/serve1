# SPDX-License-Identifier: Apache-2.0
"""Routing policies for the KaReserve gateway."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kareserve.performance import OnlineQueueTimeEstimator, PolynomialPrefillModel
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
class CostModel:
    """Convert prompt execution paths into one comparable cost unit."""

    tokens_per_work_unit: float = 256.0
    compute_ms_per_token: float | None = None
    decode_ms_per_token: float | None = None
    kv_bytes_per_token: float | None = None
    medium_profiles: dict[str, dict[str, float]] | None = None
    prefill_model: PolynomialPrefillModel | None = None
    cpu_load_weight: float = 0.25
    fs_load_weight: float = 2.0
    obj_load_weight: float = 4.0

    @property
    def unit(self) -> str:
        if self.prefill_model is not None or self.compute_ms_per_token is not None:
            return "ms"
        return "normalized"

    @classmethod
    def from_hardware_profile(
        cls,
        profile: dict[str, Any],
        *,
        tokens_per_work_unit: float,
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
        configured_decode_ms = profile.get("decode_ms_per_token")
        decode_ms = (
            float(configured_decode_ms)
            if configured_decode_ms is not None
            and float(configured_decode_ms) > 0
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
            compute_ms_per_token=compute_ms,
            decode_ms_per_token=decode_ms,
            kv_bytes_per_token=kv_bytes,
            medium_profiles=medium_profiles,
            prefill_model=PolynomialPrefillModel.from_profile(profile),
            cpu_load_weight=float(profile.get("cpu_load_weight", 0.25)),
            fs_load_weight=float(profile.get("fs_load_weight", 2.0)),
            obj_load_weight=float(profile.get("obj_load_weight", 4.0)),
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
        }
        return tokens * weights.get(medium, 0.0) / self.tokens_per_work_unit

    def prompt_path_cost(
        self,
        match: PrefixMatch,
        *,
        include_external: bool = True,
    ) -> float:
        total_tokens = match.prompt_tokens
        gpu_prefix = min(match.gpu_prefix_tokens, total_tokens)
        if not include_external:
            return self.prefill_cost(total_tokens, gpu_prefix)

        cpu_prefix = min(match.cpu_prefix_tokens, total_tokens)
        fs_prefix = min(match.fs_prefix_tokens, total_tokens)
        obj_prefix = min(match.obj_prefix_tokens, total_tokens)
        external_prefix = max(cpu_prefix, fs_prefix, obj_prefix)
        if external_prefix <= gpu_prefix:
            return self.prefill_cost(total_tokens, gpu_prefix)

        cost = self.transfer_cost(CacheMedium.CPU, external_prefix - gpu_prefix)
        memory_prefix = max(gpu_prefix, cpu_prefix)
        if external_prefix > memory_prefix:
            l2_costs = [
                self.transfer_cost(medium, external_prefix - memory_prefix)
                for medium, prefix_tokens in (
                    (CacheMedium.FS, fs_prefix),
                    (CacheMedium.OBJ, obj_prefix),
                )
                if prefix_tokens == external_prefix
            ]
            cost += min(l2_costs)
        return cost + self.prefill_cost(total_tokens, external_prefix)

    def candidate_work(self, request: SchedulerRequest, candidate: RouteCandidate) -> float:
        work = self.prompt_path_cost(candidate.prefix_match)
        if self.decode_ms_per_token is not None:
            work += request.max_tokens * self.decode_ms_per_token
        elif self.unit == "normalized":
            work += self.compute_cost(request.max_tokens)
        return work


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
    def _eligible(candidates: Iterable[RouteCandidate]) -> list[RouteCandidate]:
        values = list(candidates)
        available = [
            item for item in values
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
        return RouteAssignment(
            candidate=candidate,
            inflight_work=self.cost_model.candidate_work(request, candidate),
            estimated_cost=estimated_cost,
            cost_breakdown=cost_breakdown,
        )

    def observe_first_output(
        self,
        assignment: RouteAssignment,
        upstream_first_output_ms: float,
    ) -> None:
        return None

    def runtime_stats(self) -> dict[str, Any]:
        return {}


class TieredCompletionTimePolicy(KareserveBasePolicy):
    """Route by cache path, queue delay, and GPU KV capacity."""

    name = "tiered_completion_time"

    def __init__(
        self,
        *,
        cost_model: CostModel,
        prefix_block_size: int = 16,
        capacity_high_watermark: float = 0.80,
        capacity_hard_limit: float = 0.95,
        capacity_penalty: float = 2.0,
        queue_history_size: int = 128,
        queue_local_min_samples: int = 8,
        include_external_cache: bool = True,
    ) -> None:
        super().__init__(cost_model)
        self.prefix_block_size = max(1, prefix_block_size)
        self.capacity_high_watermark = min(max(capacity_high_watermark, 0.0), 1.0)
        self.capacity_hard_limit = min(
            max(capacity_hard_limit, self.capacity_high_watermark), 1.0
        )
        self.capacity_penalty = max(0.0, capacity_penalty)
        self.include_external_cache = include_external_cache
        self.queue_estimator = OnlineQueueTimeEstimator(
            history_size=queue_history_size,
            local_min_samples=queue_local_min_samples,
        )

    @staticmethod
    def _fits_capacity(candidate: RouteCandidate, free_blocks: int | None) -> bool:
        required = candidate.required_new_gpu_blocks
        return required is None or free_blocks is None or required <= free_blocks

    def _capacity_cost(
        self,
        candidate: RouteCandidate,
        free_blocks: int | None,
    ) -> float:
        node = candidate.node
        usage = (
            1.0 - free_blocks / node.gpu_total_blocks
            if free_blocks is not None
            and node.gpu_total_blocks is not None
            and node.gpu_total_blocks > 0
            else node.kv_cache_usage
        )
        if usage is None or usage <= self.capacity_high_watermark:
            return 0.0
        span = max(1.0 - self.capacity_high_watermark, 1e-6)
        normalized = (min(usage, 1.0) - self.capacity_high_watermark) / span
        cost = self.capacity_penalty * normalized * normalized
        if usage >= self.capacity_hard_limit:
            cost += self.capacity_penalty
        return cost

    def _prompt_path_cost(self, candidate: RouteCandidate) -> float:
        return self.cost_model.prompt_path_cost(
            candidate.prefix_match,
            include_external=self.include_external_cache,
        )

    def _queue_cost(
        self,
        candidate: RouteCandidate,
        reserved_work: float,
        prompt_path_cost: float,
    ) -> float:
        node = candidate.node
        local_prediction = self.queue_estimator.predict_ms(
            node.node_id, reserved_work
        )
        waiting = max(0, node.waiting_requests or 0)
        running = max(1, node.running_requests or 1)
        engine_prediction = waiting * prompt_path_cost / running
        return max(local_prediction, engine_prediction)

    def _breakdown(
        self,
        candidate: RouteCandidate,
        reserved_work: float,
        free_blocks: int | None,
    ) -> RouteCostBreakdown:
        prompt_path = self._prompt_path_cost(candidate)
        return RouteCostBreakdown(
            prompt_path_cost=prompt_path,
            queue_cost=self._queue_cost(candidate, reserved_work, prompt_path),
            capacity_cost=self._capacity_cost(candidate, free_blocks),
        )

    def select_batch(
        self,
        requests: list[SchedulerRequest],
        candidates: CandidateMatrix,
    ) -> dict[str, RouteAssignment]:
        reserved_work: dict[str, float] = {}
        active_requests: dict[str, int] = {}
        free_blocks: dict[str, int | None] = {}
        for by_node in candidates.values():
            for candidate in by_node.values():
                node = candidate.node
                reserved_work.setdefault(node.node_id, node.router_inflight_work)
                active_requests.setdefault(node.node_id, node.router_active_requests)
                free_blocks.setdefault(node.node_id, node.estimated_gpu_free_blocks)

        assignments: dict[str, RouteAssignment] = {}
        pending = {request.request_id: request for request in requests}
        while pending:
            request_options: dict[
                str, list[tuple[RouteCandidate, RouteCostBreakdown]]
            ] = {}
            priorities: list[tuple[int, float, str]] = []
            for request in pending.values():
                eligible = self._eligible(
                    candidates.get(request.request_id, {}).values()
                )
                fitting = [
                    item for item in eligible
                    if self._fits_capacity(
                        item, free_blocks.get(item.node.node_id)
                    )
                ]
                if fitting:
                    eligible = fitting
                if not eligible:
                    continue
                evaluated = [
                    (
                        item,
                        self._breakdown(
                            item,
                            reserved_work[item.node.node_id],
                            free_blocks.get(item.node.node_id),
                        ),
                    )
                    for item in eligible
                ]
                evaluated.sort(
                    key=lambda value: (
                        value[1].total,
                        active_requests[value[0].node.node_id],
                        value[0].node.queue_depth,
                        value[0].node.node_id,
                    )
                )
                request_options[request.request_id] = evaluated
                regret = (
                    evaluated[1][1].total - evaluated[0][1].total
                    if len(evaluated) > 1
                    else float("inf")
                )
                priorities.append((len(evaluated), -regret, request.request_id))

            if not priorities:
                break
            _, _, request_id = min(priorities)
            request = pending.pop(request_id)
            candidate, breakdown = request_options[request_id][0]
            assignment = self._assignment(
                request, candidate, breakdown.total, breakdown
            )
            assignments[request.request_id] = assignment
            node_id = candidate.node.node_id
            reserved_work[node_id] += assignment.inflight_work
            active_requests[node_id] += 1
            required = candidate.required_new_gpu_blocks
            if free_blocks.get(node_id) is not None and required is not None:
                free_blocks[node_id] = max(0, free_blocks[node_id] - required)
        return assignments

    def observe_first_output(
        self,
        assignment: RouteAssignment,
        upstream_first_output_ms: float,
    ) -> None:
        breakdown = assignment.cost_breakdown
        if breakdown is None:
            return
        observed_queue_ms = max(
            0.0, upstream_first_output_ms - breakdown.prompt_path_cost
        )
        self.queue_estimator.observe(
            assignment.candidate.node.node_id,
            assignment.candidate.node.router_inflight_work,
            observed_queue_ms,
        )

    def runtime_stats(self) -> dict[str, Any]:
        return {"queue_estimator": self.queue_estimator.stats()}


class GpuPrefixLoadPolicy(TieredCompletionTimePolicy):
    """Baseline using GPU Prefix Cache and the same load model."""

    name = "gpu_prefix_load"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["include_external_cache"] = False
        super().__init__(**kwargs)


class LeastLoadPolicy(KareserveBasePolicy):
    name = "least_load"

    def select_batch(self, requests, candidates):
        assignments: dict[str, RouteAssignment] = {}
        active: dict[str, int] = {}
        for request in requests:
            eligible = self._eligible(candidates.get(request.request_id, {}).values())
            for item in eligible:
                active.setdefault(item.node.node_id, item.node.router_active_requests)
            if not eligible:
                continue
            candidate = min(
                eligible,
                key=lambda item: (
                    active[item.node.node_id] + item.node.queue_depth,
                    item.node.node_id,
                ),
            )
            assignment = self._assignment(request, candidate, 0.0)
            assignments[request.request_id] = assignment
            active[candidate.node.node_id] += 1
        return assignments


class RoundRobinPolicy(KareserveBasePolicy):
    name = "round_robin"

    def __init__(self, cost_model: CostModel | None = None) -> None:
        super().__init__(cost_model)
        self._next_index = 0

    def select_batch(self, requests, candidates):
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

    def select_batch(self, requests, candidates):
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
