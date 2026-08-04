# SPDX-License-Identifier: Apache-2.0
"""Per-request route planning and resource reservation."""

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
        self.failed_requests = 0

    async def assign(self, request: SchedulerRequest) -> AssignmentResult:
        started_at = time.perf_counter()
        external_matches = None
        if self.lmcache_lookup is not None:
            external_matches, failed_domains = await self.lmcache_lookup.lookup([request])
            self.tracker.set_lmcache_lookup_status(failed_domains)
        lookup_completed_at = time.perf_counter()

        candidates = self.tracker.build_route_candidates(
            [request], self.prefix_block_size, external_matches
        )
        candidates_completed_at = time.perf_counter()
        assignments = self.policy.select_batch([request], candidates)
        policy_completed_at = time.perf_counter()
        assignment = assignments.get(request.request_id)
        self.total_requests += 1
        if assignment is None:
            self.failed_requests += 1
            raise RuntimeError("No available vLLM server")

        candidate = assignment.candidate
        self.tracker.reserve_route(candidate.node.node_id, assignment.inflight_work)
        return AssignmentResult(
            assignment=assignment,
            planning_total_ms=(policy_completed_at - started_at) * 1000.0,
            cache_lookup_ms=(lookup_completed_at - started_at) * 1000.0,
            candidate_build_ms=(
                candidates_completed_at - lookup_completed_at
            ) * 1000.0,
            policy_ms=(policy_completed_at - candidates_completed_at) * 1000.0,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "policy": self.policy.name,
            **self.policy.runtime_stats(),
        }
