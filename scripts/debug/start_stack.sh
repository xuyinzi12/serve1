#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
GPU_IDS=${GPU_IDS:-1}
ENABLE_LMCACHE=${KARESERVE_ENABLE_LMCACHE:-1}
ROUTER_CONFIG=${KARESERVE_CONFIG_PATH:-"$ROOT/configs/router.single-node.json"}
STARTUP_TIMEOUT_SECONDS=${KARESERVE_STARTUP_TIMEOUT_SECONDS:-120}
PROJECT_ENV="$ROOT/.venv-vllm-0.26"

"$PROJECT_ENV/bin/python" "$ROOT/scripts/experiment/validate_run.py" \
  --config "$ROUTER_CONFIG" \
  --gpu-ids $GPU_IDS

if [[ "$ENABLE_LMCACHE" == "1" ]]; then
  bash "$ROOT/scripts/debug/start_lmcache_server.sh"
  export VLLM_ENV=${VLLM_ENV:-"$PROJECT_ENV"}
  export KARESERVE_LMCACHE_MP=1
else
  export VLLM_ENV=${VLLM_ENV:-"$PROJECT_ENV"}
  export KARESERVE_LMCACHE_MP=0
fi

if [[ "$ENABLE_LMCACHE" == "1" ]]; then
  elapsed=0
  until curl -fsS "http://127.0.0.1:${LMCACHE_HTTP_PORT:-8080}/healthcheck" \
      >/dev/null 2>&1; do
    if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
      echo "LMCache did not become ready" >&2
      exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
fi

GPU_IDS="$GPU_IDS" bash "$ROOT/scripts/debug/start_vllm_cluster.sh"

wait_for_port() {
  local port="$1"
  local elapsed=0
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
      echo "vLLM on port $port did not become ready" >&2
      exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
}

for gpu in $GPU_IDS; do
  case "$gpu" in
    0) wait_for_port 8101 ;;
    1) wait_for_port 8102 ;;
    2) wait_for_port 8103 ;;
    *)
      echo "unsupported debug GPU: $gpu" >&2
      exit 2
      ;;
  esac
done

KARESERVE_CONFIG_PATH="$ROUTER_CONFIG" \
  bash "$ROOT/scripts/debug/start_router.sh"

elapsed=0
until curl -fsS "http://127.0.0.1:8090/health" >/dev/null 2>&1; do
  if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
    echo "KaReserve Router did not become ready" >&2
    exit 1
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

echo "stack ready: router=http://127.0.0.1:8090"
