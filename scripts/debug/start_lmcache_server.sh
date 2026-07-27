#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
RUNTIME="$ROOT/runtime"
ENV=${LMCACHE_ENV:-"$ROOT/.venv-vllm-0.26"}
HOST=${LMCACHE_HOST:-127.0.0.1}
PORT=${LMCACHE_PORT:-5555}
HTTP_HOST=${LMCACHE_HTTP_HOST:-127.0.0.1}
HTTP_PORT=${LMCACHE_HTTP_PORT:-8080}
L1_SIZE_GB=${LMCACHE_L1_SIZE_GB:-32}
PID_FILE="$RUNTIME/pids/lmcache-server.pid"
LOG_FILE="$RUNTIME/logs/lmcache-server.log"

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

if [[ -f "$PID_FILE" ]] &&
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "lmcache-server is already running"
  exit 0
fi

nohup "$ENV/bin/lmcache" server \
  --host "$HOST" \
  --port "$PORT" \
  --http-host "$HTTP_HOST" \
  --http-port "$HTTP_PORT" \
  --l1-size-gb "$L1_SIZE_GB" \
  --eviction-policy LRU \
  --disable-observability \
  >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"
echo "started lmcache-server pid=$!"
