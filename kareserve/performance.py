# SPDX-License-Identifier: Apache-2.0
"""Small, serializable performance models used by routing policies."""

from __future__ import annotations

import math
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
