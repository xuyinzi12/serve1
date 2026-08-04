# SPDX-License-Identifier: Apache-2.0
"""Request-level routing observations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass


def elapsed_ms(started_at: float, ended_at: float | None) -> float | None:
    if ended_at is None:
        return None
    return round(max(0.0, ended_at - started_at) * 1000.0, 3)


@dataclass(slots=True)
class RequestObservation:
    request_id: str
    trace_id: str | None
    prefix_id: str | None
    node_id: str
    request_started_at: float
    assignment_completed_at: float
    upstream_opened_at: float
    upstream_status: int
    first_output_at: float | None = None
    response_bytes: int = 0

    def result(self, outcome: str, error: str | None = None) -> dict[str, object]:
        completed_at = time.perf_counter()
        return {
            "request_id": self.request_id,
            "prefix_id": self.prefix_id,
            "trace_id": self.trace_id,
            "node_id": self.node_id,
            "outcome": outcome,
            "upstream_status": self.upstream_status,
            "assignment_ms": elapsed_ms(
                self.request_started_at, self.assignment_completed_at
            ),
            "upstream_open_ms": elapsed_ms(
                self.assignment_completed_at, self.upstream_opened_at
            ),
            "first_output_ms": elapsed_ms(
                self.request_started_at, self.first_output_at
            ),
            "total_ms": elapsed_ms(self.request_started_at, completed_at),
            "response_bytes": self.response_bytes,
            "error": error,
        }


def sse_data_has_output(data: bytes) -> bool:
    if not data or data == b"[DONE]":
        return False
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str) and text:
            return True
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return True
    return False


class SSEOutputDetector:
    """Detect the first non-empty OpenAI completion output across chunk boundaries."""

    def __init__(self) -> None:
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []

    def feed(self, chunk: bytes) -> bool:
        self._line_buffer.extend(chunk)
        while True:
            newline = self._line_buffer.find(b"\n")
            if newline < 0:
                return False
            line = bytes(self._line_buffer[:newline]).rstrip(b"\r")
            del self._line_buffer[: newline + 1]
            if line.startswith(b"data:"):
                self._event_data.append(line[5:].lstrip())
            elif not line:
                has_output = sse_data_has_output(b"\n".join(self._event_data))
                self._event_data.clear()
                if has_output:
                    return True
