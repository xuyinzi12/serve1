# SPDX-License-Identifier: Apache-2.0
"""Bounded aggregation window for cluster-level request assignment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from kareserve.kareserve_policy import KareserveBasePolicy
from kareserve.lmcache_lookup import LMCacheLookupClient
from kareserve.kareserve_state import (
    NodeRoutingState,
    PrefixMatch,
    SchedulerRequest,
)
from kareserve.kareserve_tracker import KareserveTracker


@dataclass(slots=True)
class PendingRequest:
    request: SchedulerRequest
    future: asyncio.Future[AssignmentResult]
    queued_at: float


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    node: NodeRoutingState
    prefix_match: PrefixMatch
    route_batch_size: int
    queue_wait_ms: float
    inflight_work: float
    estimated_cost: float


class RequestPool:
    def __init__(
        self,
        tracker: KareserveTracker,
        policy: KareserveBasePolicy,
        window_ms: float = 2.0,
        max_batch_size: int = 64,
        group_block_size: int = 16,
        lmcache_lookup: LMCacheLookupClient | None = None,
    ) -> None:
        self.tracker = tracker
        self.policy = policy
        self.window_seconds = max(window_ms, 0.0) / 1000.0
        self.max_batch_size = max(1, max_batch_size)
        self.group_block_size = max(1, group_block_size)
        self.lmcache_lookup = lmcache_lookup
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.total_batches = 0
        self.total_requests = 0
        self.last_batch_size = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="kareserve-request-pool")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def assign(self, request: SchedulerRequest) -> AssignmentResult:
        if self._task is None:
            self.start()
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(
            PendingRequest(
                request=request,
                future=future,
                queued_at=asyncio.get_running_loop().time(),
            )
        )
        return await future

    def stats(self) -> dict[str, Any]:
        return {
            "queued_requests": self.queue.qsize(),
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "last_batch_size": self.last_batch_size,
            "window_ms": self.window_seconds * 1000.0,
            "max_batch_size": self.max_batch_size,
        }

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.window_seconds
            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                await self._flush(batch)
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    async def _flush(self, batch: list[PendingRequest]) -> None:
        flushed_at = asyncio.get_running_loop().time()
        requests = [item.request for item in batch]
        external_matches = None
        if self.lmcache_lookup is not None:
            external_matches, failed_domains = await self.lmcache_lookup.lookup(requests)
            self.tracker.set_lmcache_lookup_status(failed_domains)
        candidates = self.tracker.build_route_candidates(
            requests, self.group_block_size, external_matches
        )
        assignments = self.policy.select_batch(requests, candidates)
        self.total_batches += 1
        self.total_requests += len(batch)
        self.last_batch_size = len(batch)

        for item in batch:
            if item.future.cancelled():
                continue
            assignment = assignments.get(item.request.request_id)
            if assignment is None:
                item.future.set_exception(RuntimeError("No available vLLM server"))
                continue
            candidate = assignment.candidate
            self.tracker.reserve_route(candidate.node.node_id, assignment.inflight_work)
            item.future.set_result(
                AssignmentResult(
                    node=candidate.node,
                    prefix_match=candidate.prefix_match,
                    route_batch_size=len(batch),
                    queue_wait_ms=max(0.0, (flushed_at - item.queued_at) * 1000.0),
                    inflight_work=assignment.inflight_work,
                    estimated_cost=assignment.estimated_cost,
                )
            )
