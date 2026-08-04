# SPDX-License-Identifier: Apache-2.0
"""Small, serializable performance models used by routing policies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PolynomialPrefillModel:
    """Second-order prefill latency model fitted from offline measurements."""

    token_scale: float
    minimum_ms: float
    intercept_ms: float
    prompt_ms: float
    cached_ms: float
    prompt_squared_ms: float
    prompt_cached_ms: float
    cached_squared_ms: float

    @classmethod
    def from_profile(
        cls, profile: dict[str, Any]
    ) -> PolynomialPrefillModel | None:
        raw = profile.get("prefill_time_model")
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("type") != "polynomial_v1":
            raise ValueError("Unsupported prefill_time_model")
        coefficients = raw.get("coefficients_ms")
        if not isinstance(coefficients, dict):
            raise ValueError("prefill_time_model requires coefficients_ms")
        required = {
            "intercept",
            "prompt",
            "cached",
            "prompt_squared",
            "prompt_cached",
            "cached_squared",
        }
        missing = required - coefficients.keys()
        if missing:
            raise ValueError(
                "prefill_time_model lacks coefficients: "
                + ", ".join(sorted(missing))
            )
        model = cls(
            token_scale=max(1.0, float(raw.get("token_scale", 1024.0))),
            minimum_ms=max(0.0, float(raw.get("minimum_ms", 0.0))),
            intercept_ms=float(coefficients.get("intercept", 0.0)),
            prompt_ms=float(coefficients.get("prompt", 0.0)),
            cached_ms=float(coefficients.get("cached", 0.0)),
            prompt_squared_ms=float(coefficients.get("prompt_squared", 0.0)),
            prompt_cached_ms=float(coefficients.get("prompt_cached", 0.0)),
            cached_squared_ms=float(coefficients.get("cached_squared", 0.0)),
        )
        if not all(
            math.isfinite(value)
            for value in (
                model.token_scale,
                model.minimum_ms,
                model.intercept_ms,
                model.prompt_ms,
                model.cached_ms,
                model.prompt_squared_ms,
                model.prompt_cached_ms,
                model.cached_squared_ms,
            )
        ):
            raise ValueError("prefill_time_model values must be finite")
        return model

    def predict_ms(self, prompt_tokens: int, cached_prefix_tokens: int) -> float:
        prompt = max(0, prompt_tokens) / self.token_scale
        cached = min(max(0, cached_prefix_tokens), max(0, prompt_tokens))
        cached /= self.token_scale
        value = (
            self.intercept_ms
            + self.prompt_ms * prompt
            + self.cached_ms * cached
            + self.prompt_squared_ms * prompt * prompt
            + self.prompt_cached_ms * prompt * cached
            + self.cached_squared_ms * cached * cached
        )
        return max(self.minimum_ms, value)


@dataclass(slots=True)
class _QueueEstimate:
    samples: deque[tuple[float, float]]

    @classmethod
    def with_capacity(cls, capacity: int) -> _QueueEstimate:
        return cls(deque(maxlen=capacity))

    @property
    def slope(self) -> float:
        denominator = sum(work * work for work, _ in self.samples)
        if denominator <= 0:
            return 0.0
        numerator = sum(work * delay for work, delay in self.samples)
        return max(0.0, numerator / denominator)

    @property
    def absolute_error_ms(self) -> float:
        if not self.samples:
            return 0.0
        slope = self.slope
        return sum(
            abs(delay - slope * work) for work, delay in self.samples
        ) / len(self.samples)


class OnlineQueueTimeEstimator:
    """Learn queue delay from routed requests without a model-specific constant."""

    def __init__(self, history_size: int = 128, local_min_samples: int = 8) -> None:
        self.history_size = max(8, int(history_size))
        self.local_min_samples = max(1, int(local_min_samples))
        self._global = _QueueEstimate.with_capacity(self.history_size)
        self._nodes: dict[str, _QueueEstimate] = {}

    def predict_ms(self, node_id: str, reserved_work: float) -> float:
        estimate = self._nodes.get(node_id)
        if estimate is None or len(estimate.samples) < self.local_min_samples:
            estimate = self._global
        if not estimate.samples:
            return 0.0
        return max(0.0, reserved_work) * estimate.slope

    def observe(
        self,
        node_id: str,
        reserved_work: float,
        observed_queue_ms: float,
    ) -> None:
        if reserved_work <= 0 or not math.isfinite(observed_queue_ms):
            return
        target = max(0.0, observed_queue_ms)
        sample = (reserved_work, target)
        self._global.samples.append(sample)
        state = self._nodes.setdefault(
            node_id, _QueueEstimate.with_capacity(self.history_size)
        )
        state.samples.append(sample)

    @staticmethod
    def _summary(state: _QueueEstimate) -> dict[str, float | int]:
        return {
            "samples": len(state.samples),
            "queue_delay_ratio": state.slope,
            "mean_absolute_error_ms": state.absolute_error_ms,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "cluster": self._summary(self._global),
            "nodes": {
                node_id: self._summary(state)
                for node_id, state in sorted(self._nodes.items())
            },
        }
