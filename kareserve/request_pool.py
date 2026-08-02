# SPDX-License-Identifier: Apache-2.0
"""Bounded aggregation window for cluster-level request assignment."""

import asyncio
from dataclasses import dataclass
from typing import Any

from kareserve.kareserve_policy import (
    KareserveBasePolicy,
    NodeRoutingState,
    SchedulerRequest,
)
from kareserve.kareserve_tracker import KareserveTracker

@dataclass
class PendingRequest:
    request: SchedulerRequest
    future: asyncio.Future["AssignmentResult"]
    queued_at: float

@dataclass(frozen=True)
class AssignmentResult:
    node: NodeRoutingState
    route_batch_size: int
    queue_wait_ms: float
    prefix_hit_blocks: int

class RequestPool:
    def __init__(
        self,
        tracker: KareserveTracker,
        policy: KareserveBasePolicy,
        window_ms: float = 2.0,
        max_batch_size: int = 64,
        group_block_size: int = 16,
    ) -> None:
        self.tracker = tracker
        self.policy = policy
        self.window_seconds = max(window_ms, 0.0) / 1000.0
        self.max_batch_size = max(1, max_batch_size)
        self.group_block_size = max(1, group_block_size)
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.total_batches = 0
        self.total_requests = 0
        self.last_batch_size = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

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
            self._flush(batch)

    def _flush(self, batch: list[PendingRequest]) -> None:
        flushed_at = asyncio.get_running_loop().time()
        requests = [item.request for item in batch]
        assignments = self.policy.select_batch(
            requests,
            self.tracker.get_routing_states(
                [req.prompt_tokens for req in requests]
            ),
        )
        self.total_batches += 1
        self.total_requests += len(batch)
        self.last_batch_size = len(batch)
        for req_idx, item in enumerate(batch):
            node = assignments.get(item.request.request_id)
            if node is None:
                item.future.set_exception(RuntimeError("No available vLLM server"))
            else:
                item.future.set_result(
                    AssignmentResult(
                        node=node,
                        route_batch_size=len(batch),
                        queue_wait_ms=max(
                            0.0, (flushed_at - item.queued_at) * 1000.0
                        ),
                        prefix_hit_blocks=node.matched_prefix_blocks[req_idx],
                    )
                )
