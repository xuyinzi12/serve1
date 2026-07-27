#!/usr/bin/env python3
"""Aggregate repeated vLLM Benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    paths = sorted(root.rglob("result.json")) if root.is_dir() else [root]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records:
        raise ValueError(f"No result.json files found under {root}")

    summary: dict[str, Any] = {
        "source": str(root),
        "runs": len(records),
        "files": [str(path) for path in paths],
        "metrics": {},
    }
    for metric in DEFAULT_METRICS:
        values = numeric_values(records, metric)
        if not values:
            continue
        summary["metrics"][metric] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "samples": len(values),
        }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
