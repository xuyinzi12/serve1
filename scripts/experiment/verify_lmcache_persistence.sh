#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
ENV="$ROOT/.venv-vllm-0.26"
GPU_ID=${GPU_IDS:-1}
VLLM_PORT=${KARESERVE_TEST_VLLM_PORT:-8102}
STARTUP_TIMEOUT_SECONDS=${KARESERVE_STARTUP_TIMEOUT_SECONDS:-120}

if [[ "$GPU_ID" != "1" ]]; then
  echo "LMCache persistence verification uses GPU1 and port 8102" >&2
  exit 2
fi

cleanup() {
  bash "$ROOT/scripts/debug/stop_debug_cluster.sh" || true
}
trap cleanup EXIT

GPU_IDS=1 \
KARESERVE_ENABLE_LMCACHE=1 \
KARESERVE_CONFIG_PATH="$ROOT/configs/router.single-node.json" \
bash "$ROOT/scripts/debug/start_stack.sh"

"$ENV/bin/python" "$ROOT/scripts/debug/lmcache_probe.py" \
  --base-url "http://127.0.0.1:$VLLM_PORT" \
  --phase store

bash "$ROOT/scripts/debug/stop_vllm_cluster.sh"

export VLLM_ENV="$ENV"
export KARESERVE_LMCACHE_MP=1
GPU_IDS=1 bash "$ROOT/scripts/debug/start_vllm_cluster.sh"

elapsed=0
until curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; do
  if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
    echo "vLLM did not become ready after restart" >&2
    exit 1
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

"$ENV/bin/python" "$ROOT/scripts/debug/lmcache_probe.py" \
  --base-url "http://127.0.0.1:$VLLM_PORT" \
  --phase retrieve
