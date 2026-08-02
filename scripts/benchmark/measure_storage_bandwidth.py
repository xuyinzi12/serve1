#!/usr/bin/env python3
"""Measure sequential storage throughput for the LMCache filesystem tier."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter


def transfer(file, size_bytes: int, block_bytes: int, write: bool) -> float:
    block = bytes(block_bytes)
    remaining = size_bytes
    started = perf_counter()
    while remaining > 0:
        length = min(block_bytes, remaining)
        if write:
            file.write(block[:length])
        else:
            data = file.read(length)
            if len(data) != length:
                raise RuntimeError("Storage benchmark encountered a short read")
        remaining -= length
    if write:
        file.flush()
        os.fsync(file.fileno())
    return perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--size-mb", type=int, default=1024)
    parser.add_argument("--block-mb", type=int, default=8)
    args = parser.parse_args()

    target = Path(args.path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    size_bytes = max(1, args.size_mb) * 1024 * 1024
    block_bytes = max(1, args.block_mb) * 1024 * 1024

    with tempfile.NamedTemporaryFile(dir=target, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        write_seconds = transfer(temporary, size_bytes, block_bytes, True)
    try:
        with temporary_path.open("rb", buffering=0) as source:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            read_seconds = transfer(source, size_bytes, block_bytes, False)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "path": str(target),
                "size_mb": args.size_mb,
                "block_mb": args.block_mb,
                "read_bandwidth_gbps": size_bytes / read_seconds / 1e9,
                "write_bandwidth_gbps": size_bytes / write_seconds / 1e9,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
