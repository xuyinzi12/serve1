#!/usr/bin/env python3
"""Run a reproducible KaReserve experiment from one JSON manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("vllm", "torch", "lmcache", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def set_value(env: dict[str, str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        env[name] = "1" if value else "0"
    elif isinstance(value, list):
        env[name] = " ".join(str(item) for item in value)
    elif value is not None:
        env[name] = str(value)


def build_environment(manifest: dict[str, Any], run_dir: Path) -> dict[str, str]:
    runtime = manifest["runtime"]
    benchmark = manifest["benchmark"]
    env = dict(os.environ)
    mappings = {
        "KARESERVE_ROOT": str(ROOT),
        "KARESERVE_CONFIG_PATH": str(resolve_path(manifest["router_config"])),
        "GPU_IDS": runtime["gpu_ids"],
        "KARESERVE_ENABLE_LMCACHE": runtime["enable_lmcache"],
        "LMCACHE_L1_SIZE_GB": runtime["lmcache_l1_size_gb"],
        "KARESERVE_MODEL": runtime["model"],
        "KARESERVE_MODEL_NAME": runtime["model_name"],
        "KARESERVE_DTYPE": runtime.get("dtype", "half"),
        "KARESERVE_GPU_MEMORY_UTILIZATION": runtime.get(
            "gpu_memory_utilization", 0.5
        ),
        "KARESERVE_DATASET_NAME": benchmark["dataset_name"],
        "KARESERVE_DATASET_PATH": benchmark.get("dataset_path"),
        "KARESERVE_NUM_PROMPTS": benchmark["num_prompts"],
        "KARESERVE_REQUEST_RATE": benchmark.get("request_rate", "inf"),
        "KARESERVE_MAX_CONCURRENCY": benchmark.get("max_concurrency", 64),
        "KARESERVE_SEED": benchmark.get("seed", 0),
        "KARESERVE_NUM_WARMUPS": benchmark.get("num_warmups", 0),
        "KARESERVE_READY_CHECK_TIMEOUT_SECONDS": benchmark.get(
            "ready_check_timeout_seconds", 0
        ),
        "KARESERVE_TRACE_CHUNK_SIZE": benchmark.get("trace_chunk_size", 16),
        "KARESERVE_TRACE_SEC_MULTIPLIER": benchmark.get(
            "trace_sec_multiplier", 1
        ),
        "KARESERVE_PREFIX_LEN": benchmark.get("prefix_len", 512),
        "KARESERVE_SUFFIX_LEN": benchmark.get("suffix_len", 64),
        "KARESERVE_NUM_PREFIXES": benchmark.get("num_prefixes", 8),
        "KARESERVE_OUTPUT_LEN": benchmark.get("output_len", 16),
        "KARESERVE_RESULT_DIR": str(run_dir),
        "KARESERVE_RESULT_FILENAME": "result.json",
        "KARESERVE_BENCH_LABEL": manifest["name"],
        "KARESERVE_POLICY_OVERRIDE": manifest.get("router_overrides", {}).get(
            "policy"
        ),
        "KARESERVE_WINDOW_MS_OVERRIDE": manifest.get(
            "router_overrides", {}
        ).get("window_ms"),
        "PYTHONHASHSEED": 0,
    }
    for name, value in mappings.items():
        set_value(env, name, value)

    dataset_path = env.get("KARESERVE_DATASET_PATH")
    if dataset_path:
        env["KARESERVE_DATASET_PATH"] = str(resolve_path(dataset_path))
    env["KARESERVE_MODEL"] = str(resolve_path(env["KARESERVE_MODEL"]))
    return env


def validate_manifest(manifest: dict[str, Any]) -> None:
    for key in ("name", "router_config", "runtime", "benchmark"):
        if key not in manifest:
            raise ValueError(f"Manifest is missing {key}")
    runtime = manifest["runtime"]
    for key in (
        "gpu_ids",
        "enable_lmcache",
        "lmcache_l1_size_gb",
        "model",
        "model_name",
    ):
        if key not in runtime:
            raise ValueError(f"Manifest runtime is missing {key}")
    benchmark = manifest["benchmark"]
    for key in ("dataset_name", "num_prompts"):
        if key not in benchmark:
            raise ValueError(f"Manifest benchmark is missing {key}")
    if benchmark["dataset_name"] in {"custom", "sharegpt", "timed_trace"}:
        dataset_path = benchmark.get("dataset_path")
        if not dataset_path or not resolve_path(dataset_path).is_file():
            raise ValueError(f"Dataset file does not exist: {dataset_path}")
    if not resolve_path(runtime["model"]).exists():
        raise ValueError(f"Model path does not exist: {runtime['model']}")


def read_router_state() -> dict[str, Any]:
    with urllib.request.urlopen(
        "http://127.0.0.1:8090/routing/state", timeout=10
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leave-running", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    repeats = int(manifest.get("repeats", 1))
    experiment_root = ROOT / "runtime" / "experiments" / manifest["name"]

    for run_index in range(1, repeats + 1):
        run_dir = experiment_root / f"run-{run_index:02d}"
        env = build_environment(manifest, run_dir)
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_revision": git_revision(),
                "manifest_path": str(manifest_path),
                "manifest": manifest,
                "resolved_environment": {
                    key: value
                    for key, value in sorted(env.items())
                    if key.startswith(("KARESERVE_", "LMCACHE_", "GPU_IDS"))
                },
                "packages": package_versions(),
            }
            (run_dir / "run-manifest.json").write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        run(
            ["bash", str(ROOT / "scripts/debug/stop_debug_cluster.sh")],
            env,
            args.dry_run,
        )
        run(
            ["bash", str(ROOT / "scripts/debug/start_stack.sh")],
            env,
            args.dry_run,
        )
        if not args.dry_run:
            time.sleep(float(manifest.get("settle_seconds", 2)))
        try:
            run(
                ["bash", str(ROOT / "scripts/benchmark/run_vllm_benchmark.sh")],
                env,
                args.dry_run,
            )
            if not args.dry_run:
                (run_dir / "router-state.json").write_text(
                    json.dumps(read_router_state(), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
        finally:
            if not args.leave_running:
                run(
                    ["bash", str(ROOT / "scripts/debug/stop_debug_cluster.sh")],
                    env,
                    args.dry_run,
                )

    if not args.dry_run:
        run(
            [
                str(ROOT / ".venv-vllm-0.26/bin/python"),
                str(ROOT / "scripts/benchmark/summarize_results.py"),
                str(experiment_root),
                "--output",
                str(experiment_root / "summary.json"),
            ],
            dict(os.environ),
            False,
        )


if __name__ == "__main__":
    main()
