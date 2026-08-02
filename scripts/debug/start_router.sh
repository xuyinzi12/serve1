#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
RUNTIME="$ROOT/runtime"
ENV="$ROOT/.venv-vllm-0.26"
CONFIG_PATH=${KARESERVE_CONFIG_PATH:-"$ROOT/configs/router.single-node.json"}
MODEL=${KARESERVE_MODEL:-/home/zn/llm_models/opt-1.3b}
CHAT_TEMPLATE=${KARESERVE_CHAT_TEMPLATE:-"$ROOT/configs/chat_templates/opt.jinja"}
ENABLE_LMCACHE=${KARESERVE_ENABLE_LMCACHE:-0}

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

if [[ -f "$RUNTIME/pids/kareserve.pid" ]] &&
    kill -0 "$(cat "$RUNTIME/pids/kareserve.pid")" 2>/dev/null; then
  echo "kareserve is already running"
  exit 0
fi

cd "$ROOT"
nohup env \
  KARESERVE_MODEL="$MODEL" \
  KARESERVE_CHAT_TEMPLATE="$CHAT_TEMPLATE" \
  KARESERVE_ENABLE_LMCACHE="$ENABLE_LMCACHE" \
  "$ENV/bin/python" -m kareserve.cli \
  --config "$CONFIG_PATH" \
  --host 127.0.0.1 \
  --port 8090 \
  >"$RUNTIME/logs/kareserve.log" 2>&1 &
echo "$!" >"$RUNTIME/pids/kareserve.pid"
echo "started kareserve pid=$!"
