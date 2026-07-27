#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zn/xyz/serve1
RUNTIME="$ROOT/runtime"
ENV="$ROOT/.venv-vllm-0.26"

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

if [[ -f "$RUNTIME/pids/kareserve.pid" ]] &&
    kill -0 "$(cat "$RUNTIME/pids/kareserve.pid")" 2>/dev/null; then
  echo "kareserve is already running"
  exit 0
fi

cd "$ROOT"
nohup "$ENV/bin/python" -m kareserve.cli \
  --config "$RUNTIME/config/kareserve.json" \
  --host 127.0.0.1 \
  --port 8090 \
  >"$RUNTIME/logs/kareserve.log" 2>&1 &
echo "$!" >"$RUNTIME/pids/kareserve.pid"
echo "started kareserve pid=$!"
