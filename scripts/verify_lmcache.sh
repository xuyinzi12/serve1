#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
ENV="$ROOT/.venv-vllm-0.26"
GPU_IDS=${GPU_IDS:-0,3}
ROUTER_URL=${KARESERVE_BENCH_BASE_URL:-http://127.0.0.1:8090}

cleanup() {
  bash "$ROOT/scripts/stop.sh" || true
}
trap cleanup EXIT

bash "$ROOT/scripts/stop.sh"

stop_compute_plane() {
  local pid_file pid
  for pid_file in "$ROOT/runtime"/pids/vllm-*.pid \
                  "$ROOT/runtime/pids/router.pid"; do
    [[ -e "$pid_file" ]] || continue
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    rm -f -- "$pid_file"
  done
}

GPU_IDS="$GPU_IDS" KARESERVE_ENABLE_LMCACHE=1 \
  KARESERVE_POLICY_OVERRIDE=tiered_completion_time \
  bash "$ROOT/scripts/start.sh"

"$ENV/bin/python" "$ROOT/scripts/lmcache_probe.py" \
  --base-url "$ROUTER_URL" --metrics-url http://127.0.0.1:8101 \
  --phase store --stream

stop_compute_plane

GPU_IDS="$GPU_IDS" KARESERVE_ENABLE_LMCACHE=1 \
  KARESERVE_POLICY_OVERRIDE=tiered_completion_time \
  bash "$ROOT/scripts/start.sh"

"$ENV/bin/python" "$ROOT/scripts/lmcache_probe.py" \
  --base-url "$ROUTER_URL" --metrics-url http://127.0.0.1:8101 \
  --phase retrieve --stream
