# SPDX-License-Identifier: Apache-2.0
"""Event-driven request coordination for joint route planning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from kareserve.routing import AssignmentResult, RoutePlanner
from kareserve.state import SchedulerRequest


@dataclass(slots=True)
class _PendingRequest:
    request: SchedulerRequest
    future: asyncio.Future[AssignmentResult]
    queued_at: float


class RequestPool:
    """Group requests that are already ready without a fixed time window."""

    def __init__(
        self,
        planner: RoutePlanner,
        max_planning_group_size: int = 64,
    ) -> None:
        self.planner = planner
        self.max_planning_group_size = max(1, max_planning_group_size)
        self.queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.total_groups = 0
        self.total_requests = 0
        self.largest_group = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name="kareserve-request-pool"
            )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def assign(self, request: SchedulerRequest) -> AssignmentResult:
        if self._task is None:
            self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AssignmentResult] = loop.create_future()
        await self.queue.put(_PendingRequest(request, future, loop.time()))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            await asyncio.sleep(0)
            group = [first]
            while len(group) < self.max_planning_group_size:
                try:
                    group.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            planning_started_at = asyncio.get_running_loop().time()
            try:
                results = await self.planner.plan(
                    [item.request for item in group]
                )
            except Exception as exc:  # noqa: BLE001
                for item in group:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            self.total_groups += 1
            self.total_requests += len(group)
            self.largest_group = max(self.largest_group, len(group))
            for item in group:
                if item.future.cancelled():
                    result = results[item.request.request_id]
                    self.planner.tracker.release_route(
                        result.node.node_id, result.inflight_work
                    )
                    continue
                result = results[item.request.request_id]
                item.future.set_result(
                    replace(
                        result,
                        coordination_wait_ms=max(
                            0.0,
                            (planning_started_at - item.queued_at) * 1000.0,
                        ),
                    )
                )

    def stats(self) -> dict[str, Any]:
        mean_group_size = (
            self.total_requests / self.total_groups
            if self.total_groups
            else 0.0
        )
        return {
            "queued_requests": self.queue.qsize(),
            "total_groups": self.total_groups,
            "total_requests": self.total_requests,
            "mean_group_size": mean_group_size,
            "largest_group": self.largest_group,
            "max_planning_group_size": self.max_planning_group_size,
        }
