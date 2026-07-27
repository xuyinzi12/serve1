#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
RUNTIME="$ROOT/runtime"
ENV=${VLLM_ENV:-"$ROOT/.venv-vllm-0.26"}
CUDA_HOME=${VLLM_CUDA_HOME:-"$ENV/lib/python3.12/site-packages/nvidia/cu13"}
if [[ ! -d "$CUDA_HOME" ]]; then
  CUDA_HOME="$ROOT/.venv-vllm-0.26/lib/python3.12/site-packages/nvidia/cu13"
fi
MODEL=${KARESERVE_MODEL:-/home/zn/llm_models/opt-1.3b}
MODEL_NAME=${KARESERVE_MODEL_NAME:-kareserve-opt-1.3b}
DTYPE=${KARESERVE_DTYPE:-half}
GPU_MEMORY_UTILIZATION=${KARESERVE_GPU_MEMORY_UTILIZATION:-0.5}
CHAT_TEMPLATE=${KARESERVE_CHAT_TEMPLATE-"$ROOT/configs/chat_templates/opt.jinja"}

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

start_vllm() {
  local gpu="$1"
  local http_port="$2"
  local event_port="$3"
  local name="$4"
  local -a kv_args=()
  local -a chat_args=()

  if [[ "${KARESERVE_LMCACHE_MP:-0}" == "1" ]]; then
    kv_args=(
      --kv-offloading-size "${KARESERVE_LMCACHE_TRIGGER_GB:-1}"
      --kv-offloading-backend lmcache
      --disable-hybrid-kv-cache-manager
    )
  fi
  if [[ -n "$CHAT_TEMPLATE" ]]; then
    chat_args=(--chat-template "$CHAT_TEMPLATE")
  fi

  if [[ -f "$RUNTIME/pids/$name.pid" ]] &&
      kill -0 "$(cat "$RUNTIME/pids/$name.pid")" 2>/dev/null; then
    echo "$name is already running"
    return
  fi

  nohup env -u VLLM_ENV -u VLLM_CUDA_HOME \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA_HOME="$CUDA_HOME" \
    CUDACXX="$CUDA_HOME/bin/nvcc" \
    PATH="$CUDA_HOME/bin:$PATH" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$ENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$http_port" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    "${chat_args[@]}" \
    "${kv_args[@]}" \
    --kv-events-config \
    "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$event_port\"}" \
    >"$RUNTIME/logs/$name.log" 2>&1 &
  echo "$!" >"$RUNTIME/pids/$name.pid"
  echo "started $name pid=$!"
}

for gpu in ${GPU_IDS:-0 1}; do
  case "$gpu" in
    0) start_vllm 0 8101 5557 vllm-gpu0 ;;
    1) start_vllm 1 8102 5558 vllm-gpu1 ;;
    2) start_vllm 2 8103 5559 vllm-gpu2 ;;
    *)
      echo "unsupported debug GPU: $gpu" >&2
      exit 2
      ;;
  esac
done
