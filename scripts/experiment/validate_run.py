#!/usr/bin/env python3
"""Validate the debug GPU-to-port mapping against a Router configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


GPU_PORTS = {
    0: (8101, 5557),
    1: (8102, 5558),
    2: (8103, 5559),
}


def endpoint_port(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    if parsed.port is None:
        raise ValueError(f"KV Event endpoint has no port: {endpoint}")
    return parsed.port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("GPU IDs must be unique")
    unsupported = [gpu for gpu in args.gpu_ids if gpu not in GPU_PORTS]
    if unsupported:
        raise ValueError(f"Unsupported debug GPU IDs: {unsupported}")

    nodes = config.get("nodes", [])
    if len(nodes) != len(args.gpu_ids):
        raise ValueError(
            f"Router has {len(nodes)} nodes while GPU_IDS has "
            f"{len(args.gpu_ids)} entries"
        )

    expected = {GPU_PORTS[gpu] for gpu in args.gpu_ids}
    actual = {
        (int(node["port"]), endpoint_port(node["kv_events_endpoint"]))
        for node in nodes
    }
    if actual != expected:
        raise ValueError(
            f"Router ports {sorted(actual)} do not match GPU ports "
            f"{sorted(expected)}"
        )

    node_ids = [str(node["node_id"]) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Router node IDs must be unique")
    if config.get("tokenizer_node_id") not in node_ids:
        raise ValueError("tokenizer_node_id must reference a configured node")

    print(
        json.dumps(
            {
                "config": str(config_path),
                "gpu_ids": args.gpu_ids,
                "nodes": node_ids,
                "tokenizer_node_id": config["tokenizer_node_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
