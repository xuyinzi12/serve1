# SPDX-License-Identifier: Apache-2.0
"""Joint route planning and atomic resource reservation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kareserve.lmcache_client import LMCacheLookupClient
from kareserve.policy import KareserveBasePolicy
from kareserve.state import (
    NodeRoutingState,
    PrefixMatch,
    RouteAssignment,
    RouteCostBreakdown,
    SchedulerRequest,
)
from kareserve.tracker import KareserveTracker


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    assignment: RouteAssignment
    planning_group_size: int
    coordination_wait_ms: float
    planning_total_ms: float
    cache_lookup_ms: float
    candidate_build_ms: float
    policy_ms: float

    @property
    def node(self) -> NodeRoutingState:
        return self.assignment.candidate.node

    @property
    def prefix_match(self) -> PrefixMatch:
        return self.assignment.candidate.prefix_match

    @property
    def inflight_work(self) -> float:
        return self.assignment.inflight_work

    @property
    def estimated_cost(self) -> float:
        return self.assignment.estimated_cost

    @property
    def cost_breakdown(self) -> RouteCostBreakdown | None:
        return self.assignment.cost_breakdown


class RoutePlanner:
    def __init__(
        self,
        tracker: KareserveTracker,
        policy: KareserveBasePolicy,
        prefix_block_size: int = 16,
        lmcache_lookup: LMCacheLookupClient | None = None,
    ) -> None:
        self.tracker = tracker
        self.policy = policy
        self.prefix_block_size = max(1, prefix_block_size)
        self.lmcache_lookup = lmcache_lookup
        self.total_requests = 0
        self.total_plans = 0
        self.failed_requests = 0

    async def assign(self, request: SchedulerRequest) -> AssignmentResult:
        return (await self.plan([request]))[request.request_id]

    async def plan(
        self, requests: list[SchedulerRequest]
    ) -> dict[str, AssignmentResult]:
        if not requests:
            return {}
        started_at = time.perf_counter()
        external_matches = None
        if self.lmcache_lookup is not None:
            external_matches, failed_domains = await self.lmcache_lookup.lookup(requests)
            self.tracker.set_lmcache_lookup_status(failed_domains)
        lookup_completed_at = time.perf_counter()

        candidates = self.tracker.build_route_candidates(
            requests, self.prefix_block_size, external_matches
        )
        candidates_completed_at = time.perf_counter()
        assignments = self.policy.select_batch(requests, candidates)
        policy_completed_at = time.perf_counter()
        missing = [
            request.request_id
            for request in requests
            if request.request_id not in assignments
        ]
        self.total_requests += len(requests)
        self.total_plans += 1
        if missing:
            self.failed_requests += len(missing)
            raise RuntimeError("No available vLLM server")

        for assignment in assignments.values():
            candidate = assignment.candidate
            self.tracker.reserve_route(
                candidate.node.node_id, assignment.inflight_work
            )

        planning_total_ms = (policy_completed_at - started_at) * 1000.0
        cache_lookup_ms = (lookup_completed_at - started_at) * 1000.0
        candidate_build_ms = (
            candidates_completed_at - lookup_completed_at
        ) * 1000.0
        policy_ms = (policy_completed_at - candidates_completed_at) * 1000.0
        return {
            request.request_id: AssignmentResult(
                assignment=assignments[request.request_id],
                planning_group_size=len(requests),
                coordination_wait_ms=0.0,
                planning_total_ms=planning_total_ms,
                cache_lookup_ms=cache_lookup_ms,
                candidate_build_ms=candidate_build_ms,
                policy_ms=policy_ms,
            )
            for request in requests
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_plans": self.total_plans,
            "failed_requests": self.failed_requests,
            "policy": self.policy.name,
            **self.policy.runtime_stats(),
        }
