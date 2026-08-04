#!/usr/bin/env python3
"""Send one deterministic long-prefix request and report cache metrics."""

import argparse
import json
import re
import time
import urllib.request


METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)


def get_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for metric in METRICS:
        matches = re.findall(
            rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
            text,
            re.MULTILINE,
        )
        values[metric] = sum(float(value) for value in matches)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8102")
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="vLLM metrics base URL; defaults to --base-url",
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    metrics_url = args.metrics_url or args.base_url

    shared_prefix = (
        "LMCache shared prefix validation context contains deterministic "
        "tokens for external KV cache storage and retrieval. "
    ) * 96
    body = {
        "model": "kareserve-opt-1.3b",
        "prompt": f"{shared_prefix}\nQuestion: return one word.\nAnswer:",
        "max_tokens": 8,
        "temperature": 0,
        "stream": args.stream,
    }
    request = urllib.request.Request(
        f"{args.base_url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        response_headers = {
            key.lower(): value for key, value in response.headers.items()
        }
        first_output_ms = None
        completion_id = None
        if args.stream:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                completion_id = completion_id or payload.get("id")
                choices = payload.get("choices", [])
                if first_output_ms is None and any(
                    choice.get("text")
                    or (
                        isinstance(choice.get("delta"), dict)
                        and choice["delta"].get("content")
                    )
                    for choice in choices
                ):
                    first_output_ms = (time.perf_counter() - started) * 1000.0
        else:
            response_body = json.loads(response.read().decode("utf-8"))
            completion_id = response_body.get("id")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "phase": args.phase,
                "elapsed_ms": round(elapsed_ms, 3),
                "first_output_ms": (
                    round(first_output_ms, 3)
                    if first_output_ms is not None
                    else None
                ),
                "completion_id": completion_id,
                "route": {
                    key: value
                    for key, value in response_headers.items()
                    if key.startswith("x-kareserve-")
                },
                "metrics": get_metrics(metrics_url),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
