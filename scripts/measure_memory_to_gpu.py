#!/usr/bin/env python3
"""Measure pinned host-memory to GPU transfer latency and bandwidth."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

import torch


def parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive MiB values")
    return sizes


def measure_size(
    device: torch.device,
    size_mib: int,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    size_bytes = size_mib * 1024 * 1024
    source = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    target = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            target.copy_(source, non_blocking=True)
    stream.synchronize()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            target.copy_(source, non_blocking=True)
            end.record(stream)
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    median_ms = statistics.median(samples_ms)
    bandwidth_gbps = size_bytes / (median_ms / 1000.0) / 1e9
    return {
        "size_mib": float(size_mib),
        "median_ms": median_ms,
        "p10_ms": sorted(samples_ms)[max(0, iterations // 10 - 1)],
        "p90_ms": sorted(samples_ms)[min(iterations - 1, iterations * 9 // 10)],
        "bandwidth_gbps": bandwidth_gbps,
    }


def linear_profile(results: list[dict[str, float]]) -> tuple[float, float]:
    points = [
        (result["size_mib"] * 1024 * 1024, result["median_ms"] / 1000.0)
        for result in results
    ]
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        return max(results[0]["bandwidth_gbps"], 0.0), 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    bandwidth_gbps = 1.0 / slope / 1e9 if slope > 0 else 0.0
    return bandwidth_gbps, max(0.0, intercept * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes-mib", type=parse_sizes, default=parse_sizes("1,4,16,64"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device {args.device} is unavailable")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations must be positive")

    device = torch.device(f"cuda:{args.device}")
    properties = torch.cuda.get_device_properties(device)
    results = [
        measure_size(device, size, args.warmup, args.iterations)
        for size in args.sizes_mib
    ]
    fitted_bandwidth, fitted_latency = linear_profile(results)
    output: dict[str, Any] = {
        "device_index": args.device,
        "device_name": properties.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "memory_total_bytes": properties.total_memory,
        "transfer": "host_memory_to_gpu",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": results,
        "profile": {
            "bandwidth_gbps": fitted_bandwidth,
            "base_latency_ms": fitted_latency,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
