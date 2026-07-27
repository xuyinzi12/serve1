#!/usr/bin/env python3
"""Build a deterministic vLLM timed_trace workload with repeated prefixes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=100.0)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_requests < args.num_prefixes:
        raise ValueError("num-requests must be at least num-prefixes")
    if args.request_rate <= 0 or args.burstiness <= 0:
        raise ValueError("request-rate and burstiness must be positive")
    if min(args.prefix_len, args.suffix_len, args.output_len) < 0:
        raise ValueError("token lengths must be non-negative")

    rng = random.Random(args.seed)
    prefix_assignments = [
        index % args.num_prefixes for index in range(args.num_requests)
    ]
    rng.shuffle(prefix_assignments)

    prefix_chunks = math.ceil(args.prefix_len / args.chunk_size)
    suffix_chunks = math.ceil(args.suffix_len / args.chunk_size)
    shared = {
        prefix_id: [
            1_000_000 + prefix_id * 10_000 + index
            for index in range(prefix_chunks)
        ]
        for prefix_id in range(args.num_prefixes)
    }

    timestamps = [0.0]
    for _ in range(1, args.num_requests):
        interval = rng.gammavariate(
            args.burstiness,
            1.0 / (args.request_rate * args.burstiness),
        )
        timestamps.append(timestamps[-1] + interval)
    if len(timestamps) > 1 and timestamps[-1] > 0:
        target_duration = (args.num_requests - 1) / args.request_rate
        scale = target_duration / timestamps[-1]
        timestamps = [value * scale for value in timestamps]

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for request_index, prefix_id in enumerate(prefix_assignments):
            suffix = [
                2_000_000 + request_index * 10_000 + index
                for index in range(suffix_chunks)
            ]
            file.write(
                json.dumps(
                    {
                        "timestamp": round(timestamps[request_index], 9),
                        "input_length": args.prefix_len + args.suffix_len,
                        "output_length": args.output_len,
                        "hash_ids": shared[prefix_id] + suffix,
                        "prefix_id": prefix_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "requests": args.num_requests,
                "prefixes": args.num_prefixes,
                "duration_seconds": round(timestamps[-1], 6),
                "chunk_size": args.chunk_size,
                "seed": args.seed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
