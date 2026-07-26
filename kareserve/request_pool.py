# SPDX-License-Identifier: Apache-2.0
"""Bounded aggregation window for cluster-level request assignment."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List

from kareserve.kareserve_policy import (
    KareserveBasePolicy,
    NodeState,
    SchedulerRequest,
)
from kareserve.kareserve_tracker import KareserveTracker


@dataclass
class PendingRequest:
    request: SchedulerRequest
    future: asyncio.Future[NodeState]


class RequestPool:
    def __init__(
        self,
        tracker: KareserveTracker,
        policy: KareserveBasePolicy,
        window_ms: float = 2.0,
        max_batch_size: int = 64,
    ) -> None:
        self.tracker = tracker
        self.policy = policy
        self.window_seconds = max(window_ms, 0.0) / 1000.0
        self.max_batch_size = max(1, max_batch_size)
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

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

    async def assign(self, request: SchedulerRequest) -> NodeState:
        if self._task is None:
            self.start()
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(PendingRequest(request, future))
        return await future

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

    def _flush(self, batch: List[PendingRequest]) -> None:
        requests = [item.request for item in batch]
        assignments = self.policy.select_batch(
            requests, self.tracker.get_node_states()
        )
        for item in batch:
            node = assignments.get(item.request.request_id)
            if node is None:
                item.future.set_exception(RuntimeError("No available vLLM server"))
            else:
                item.future.set_result(node)
