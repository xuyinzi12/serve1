#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
MODE=${1:-stack}
if [[ "$MODE" != "stack" && "$MODE" != "vllm" ]]; then
  echo "usage: $0 [stack|vllm]" >&2
  exit 2
fi
RUNTIME="$ROOT/runtime"
ENV="$ROOT/.venv-vllm-0.26"
CONFIG=${KARESERVE_CONFIG_PATH:-"$ROOT/configs/config.json"}
GPU_IDS=${GPU_IDS:-1}
ENABLE_LMCACHE=${KARESERVE_ENABLE_LMCACHE:-1}
STARTUP_TIMEOUT=${KARESERVE_STARTUP_TIMEOUT_SECONDS:-120}
MODEL=${KARESERVE_MODEL:-/home/zn/llm_models/opt-1.3b}
MODEL_NAME=${KARESERVE_MODEL_NAME:-kareserve-opt-1.3b}
DTYPE=${KARESERVE_DTYPE:-half}
GPU_MEMORY_UTILIZATION=${KARESERVE_GPU_MEMORY_UTILIZATION:-0.5}
CHAT_TEMPLATE=${KARESERVE_CHAT_TEMPLATE:-}
CUDA_HOME=${VLLM_CUDA_HOME:-"$ENV/lib/python3.12/site-packages/nvidia/cu13"}

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"
"$ENV/bin/python" "$ROOT/scripts/validate_config.py" \
  --config "$CONFIG" --gpu-ids $GPU_IDS

started=0
cleanup_on_error() {
  if [[ "$started" != "1" ]]; then
    bash "$ROOT/scripts/stop.sh" || true
  fi
}
trap cleanup_on_error EXIT

start_lmcache() {
  local pid_file="$RUNTIME/pids/lmcache.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    return
  fi

  local l2_path=${LMCACHE_L2_PATH:-"$RUNTIME/lmcache/l2"}
  local -a args=(
    --host "${LMCACHE_HOST:-127.0.0.1}"
    --port "${LMCACHE_PORT:-5555}"
    --http-host "${LMCACHE_HTTP_HOST:-127.0.0.1}"
    --http-port "${LMCACHE_HTTP_PORT:-8080}"
    --l1-size-gb "${LMCACHE_L1_SIZE_GB:-8}"
    --eviction-policy LRU
    --disable-observability
  )
  if [[ "${LMCACHE_ENABLE_L2:-1}" == "1" ]]; then
    mkdir -p "$l2_path"
    args+=(
      --l2-adapter
      "{\"type\":\"fs_native\",\"base_path\":\"$l2_path\",\"max_capacity_gb\":${LMCACHE_L2_CAPACITY_GB:-64},\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}"
    )
  fi
  nohup "$ENV/bin/python" -m kareserve.lmcache_server "${args[@]}" \
    >"$RUNTIME/logs/lmcache.log" 2>&1 &
  echo "$!" >"$pid_file"
}

start_vllm() {
  local gpu="$1" http_port="$2" event_port="$3" replay_port="$4"
  local name="vllm-gpu$gpu"
  local pid_file="$RUNTIME/pids/$name.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    return
  fi

  local -a chat_args=() lmcache_args=() cache_args=()
  [[ -z "$CHAT_TEMPLATE" ]] || chat_args=(--chat-template "$CHAT_TEMPLATE")
  if [[ "$ENABLE_LMCACHE" == "1" ]]; then
    lmcache_args=(
      --kv-offloading-size "${KARESERVE_LMCACHE_TRIGGER_GB:-1}"
      --kv-offloading-backend lmcache
      --disable-hybrid-kv-cache-manager
    )
  fi
  if [[ -n "${KARESERVE_NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
    cache_args=(--num-gpu-blocks-override "$KARESERVE_NUM_GPU_BLOCKS_OVERRIDE")
  fi

  nohup env -u VLLM_ENV -u VLLM_CUDA_HOME \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA_HOME="$CUDA_HOME" \
    CUDACXX="$CUDA_HOME/bin/nvcc" \
    PATH="$CUDA_HOME/bin:$PATH" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$ENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$http_port" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    "${chat_args[@]}" "${lmcache_args[@]}" "${cache_args[@]}" \
    --kv-events-config \
    "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$event_port\",\"replay_endpoint\":\"tcp://*:$replay_port\"}" \
    >"$RUNTIME/logs/$name.log" 2>&1 &
  echo "$!" >"$pid_file"
}

wait_for_health() {
  local url="$1" name="$2" elapsed=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( elapsed >= STARTUP_TIMEOUT )); then
      echo "$name did not become ready" >&2
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
}

if [[ "$MODE" == "stack" && "$ENABLE_LMCACHE" == "1" ]]; then
  start_lmcache
  wait_for_health "http://127.0.0.1:${LMCACHE_HTTP_PORT:-8080}/healthcheck" LMCache
fi

for gpu in $GPU_IDS; do
  case "$gpu" in
    0) start_vllm 0 8101 5557 6557 ;;
    1) start_vllm 1 8102 5558 6558 ;;
    2) start_vllm 2 8103 5559 6559 ;;
  esac
done
for gpu in $GPU_IDS; do
  case "$gpu" in
    0) wait_for_health http://127.0.0.1:8101/health vLLM-GPU0 ;;
    1) wait_for_health http://127.0.0.1:8102/health vLLM-GPU1 ;;
    2) wait_for_health http://127.0.0.1:8103/health vLLM-GPU2 ;;
  esac
done

if [[ "$MODE" == "stack" ]]; then
  router_pid_file="$RUNTIME/pids/router.pid"
  if [[ ! -f "$router_pid_file" ]] || \
      ! kill -0 "$(cat "$router_pid_file")" 2>/dev/null; then
    nohup env \
      KARESERVE_MODEL="$MODEL" \
      KARESERVE_CHAT_TEMPLATE="$CHAT_TEMPLATE" \
      KARESERVE_ENABLE_LMCACHE="$ENABLE_LMCACHE" \
      "$ENV/bin/python" -m kareserve.cli \
      --config "$CONFIG" --host 127.0.0.1 --port 8090 \
      >"$RUNTIME/logs/router.log" 2>&1 &
    echo "$!" >"$router_pid_file"
  fi
  wait_for_health http://127.0.0.1:8090/health Router
fi

started=1
if [[ "$MODE" == "stack" ]]; then
  echo "stack ready: http://127.0.0.1:8090"
else
  echo "vLLM ready"
fi
