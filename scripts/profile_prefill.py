# SPDX-License-Identifier: Apache-2.0
"""Fit the Router prefill latency model against an idle vLLM instance."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from transformers import AutoTokenizer


FEATURE_NAMES = (
    "intercept",
    "prompt",
    "cached",
    "prompt_squared",
    "prompt_cached",
    "cached_squared",
)

TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
LOCAL_HIT = "vllm:prompt_tokens_by_source_total"


def parse_numbers(value: str, cast: type) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def make_tokens(length: int, salt: int, vocab_size: int) -> list[int]:
    usable = max(vocab_size - 256, 1)
    value = (salt + 1) & 0x7FFFFFFF
    tokens: list[int] = []
    for _ in range(length):
        value = (1103515245 * value + 12345) & 0x7FFFFFFF
        tokens.append(128 + value % usable)
    return tokens


async def completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    tokens: list[int],
    *,
    stream: bool,
) -> float:
    body = {
        "model": model,
        "prompt": tokens,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
    }
    started = time.perf_counter()
    async with session.post(f"{endpoint}/v1/completions", json=body) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(f"vLLM returned HTTP {response.status}: {detail}")
        if not stream:
            await response.read()
            return (time.perf_counter() - started) * 1000.0
        first_token_ms: float | None = None
        async for chunk in response.content:
            for line in chunk.splitlines():
                if (
                    first_token_ms is None
                    and line.startswith(b"data:")
                    and b"[DONE]" not in line
                ):
                    first_token_ms = (time.perf_counter() - started) * 1000.0
        if first_token_ms is not None:
            return first_token_ms
    raise RuntimeError("vLLM stream ended before the first token")


def metric_value(text: str, name: str, *, source: str | None = None) -> float:
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        if source is not None and f'source="{source}"' not in line:
            continue
        match = re.search(r"\s([-+0-9.eE]+)$", line)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"vLLM metric is unavailable: {name}")


async def metrics_snapshot(
    session: aiohttp.ClientSession, endpoint: str
) -> tuple[float, float, float]:
    async with session.get(f"{endpoint}/metrics") as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM metrics returned HTTP {response.status}")
        text = await response.text()
    return (
        metric_value(text, TTFT_COUNT),
        metric_value(text, TTFT_SUM),
        metric_value(text, LOCAL_HIT, source="local_cache_hit"),
    )


async def collect(args: argparse.Namespace) -> list[dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = int(tokenizer.vocab_size)
    lengths = parse_numbers(args.prompt_lengths, int)
    ratios = parse_numbers(args.prefix_ratios, float)
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("prompt lengths must be positive")
    if not ratios or any(ratio < 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("prefix ratios must be in [0, 1)")
    if args.block_size <= 0 or args.trials <= 0:
        raise ValueError("block size and trials must be positive")

    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    samples: list[dict[str, float]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await completion(
            session,
            args.endpoint,
            args.model,
            make_tokens(min(lengths), 0, vocab_size),
            stream=False,
        )
        sample_id = 1
        for prompt_tokens in lengths:
            await completion(
                session,
                args.endpoint,
                args.model,
                make_tokens(prompt_tokens, sample_id * 100_000, vocab_size),
                stream=False,
            )
            for ratio in ratios:
                cached_tokens = int(prompt_tokens * ratio)
                cached_tokens -= cached_tokens % args.block_size
                cached_tokens = min(cached_tokens, prompt_tokens - 1)
                timings: list[float] = []
                observed_hits: list[float] = []
                for trial in range(args.trials):
                    salt = sample_id * 1000 + trial
                    prefix = make_tokens(cached_tokens, salt, vocab_size)
                    if prefix:
                        await completion(
                            session,
                            args.endpoint,
                            args.model,
                            prefix,
                            stream=False,
                        )
                        await asyncio.sleep(args.cache_settle_seconds)
                    suffix = make_tokens(
                        prompt_tokens - cached_tokens,
                        salt + 10_000_000,
                        vocab_size,
                    )
                    before_count, before_ttft, before_hits = await metrics_snapshot(
                        session, args.endpoint
                    )
                    await completion(
                        session,
                        args.endpoint,
                        args.model,
                        prefix + suffix,
                        stream=True,
                    )
                    after_count, after_ttft, after_hits = await metrics_snapshot(
                        session, args.endpoint
                    )
                    if round(after_count - before_count) != 1:
                        raise RuntimeError(
                            "vLLM received concurrent requests during profiling"
                        )
                    timings.append((after_ttft - before_ttft) * 1000.0)
                    observed_hits.append(after_hits - before_hits)
                samples.append(
                    {
                        "prompt_tokens": float(prompt_tokens),
                        "cached_prefix_tokens": statistics.median(observed_hits),
                        "ttft_ms": statistics.median(timings),
                    }
                )
                sample_id += 1
    return samples


def fit(samples: list[dict[str, float]], token_scale: float) -> dict[str, Any]:
    if len(samples) < len(FEATURE_NAMES):
        raise ValueError("at least six profile points are required")
    rows = []
    observed = []
    for sample in samples:
        prompt = sample["prompt_tokens"] / token_scale
        cached = sample["cached_prefix_tokens"] / token_scale
        rows.append((1.0, prompt, cached, prompt * prompt, prompt * cached, cached * cached))
        observed.append(sample["ttft_ms"])
    coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        rcond=None,
    )
    predicted = np.asarray(rows, dtype=np.float64) @ coefficients
    residual = predicted - np.asarray(observed, dtype=np.float64)
    return {
        "type": "polynomial_v1",
        "token_scale": token_scale,
        "minimum_ms": float(np.percentile(observed, 5)),
        "coefficients_ms": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, coefficients, strict=True)
        },
        "fit": {
            "sample_count": len(samples),
            "rmse_ms": float(np.sqrt(np.mean(residual * residual))),
            "max_absolute_error_ms": float(np.max(np.abs(residual))),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8101")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--prompt-lengths",
        default="64,128,256,512,1024,1536,1920",
    )
    parser.add_argument("--prefix-ratios", default="0,0.25,0.5,0.75")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cache-settle-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-config", type=Path)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    samples = await collect(args)
    model = fit(samples, max(sample["prompt_tokens"] for sample in samples))
    result = {"prefill_time_model": model, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.update_config is not None:
        config = json.loads(args.update_config.read_text(encoding="utf-8"))
        config.setdefault("hardware_profile", {})["prefill_time_model"] = model
        args.update_config.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
