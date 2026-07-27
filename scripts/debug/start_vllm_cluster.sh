#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zn/xyz/serve1
RUNTIME="$ROOT/runtime"
ENV="$ROOT/.venv-vllm-0.26"
MODEL=/home/zn/llm_models/opt-1.3b
MODEL_NAME=kareserve-opt-1.3b

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

start_vllm() {
  local gpu="$1"
  local http_port="$2"
  local event_port="$3"
  local name="$4"

  if [[ -f "$RUNTIME/pids/$name.pid" ]] &&
      kill -0 "$(cat "$RUNTIME/pids/$name.pid")" 2>/dev/null; then
    echo "$name is already running"
    return
  fi

  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$ENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$http_port" \
    --dtype half \
    --gpu-memory-utilization 0.5 \
    --enable-prefix-caching \
    --chat-template "$ROOT/examples/opt_chat_template.jinja" \
    --kv-events-config \
    "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$event_port\"}" \
    >"$RUNTIME/logs/$name.log" 2>&1 &
  echo "$!" >"$RUNTIME/pids/$name.pid"
  echo "started $name pid=$!"
}

start_vllm 0 8101 5557 vllm-gpu0
start_vllm 1 8102 5558 vllm-gpu1
