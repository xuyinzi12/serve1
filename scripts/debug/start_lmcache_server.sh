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
ENABLE_L2=${LMCACHE_ENABLE_L2:-1}
L2_PATH=${LMCACHE_L2_PATH:-"$RUNTIME/lmcache/l2"}
L2_CAPACITY_GB=${LMCACHE_L2_CAPACITY_GB:-64}
PID_FILE="$RUNTIME/pids/lmcache-server.pid"
LOG_FILE="$RUNTIME/logs/lmcache-server.log"

mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

if [[ -f "$PID_FILE" ]] &&
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "lmcache-server is already running"
  exit 0
fi

server_args=(
  --host "$HOST"
  --port "$PORT"
  --http-host "$HTTP_HOST"
  --http-port "$HTTP_PORT"
  --l1-size-gb "$L1_SIZE_GB"
  --eviction-policy LRU
  --disable-observability
)

if [[ "$ENABLE_L2" == "1" ]]; then
  mkdir -p "$L2_PATH"
  server_args+=(
    --l2-adapter
    "{\"type\":\"fs_native\",\"base_path\":\"$L2_PATH\",\"max_capacity_gb\":$L2_CAPACITY_GB,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}"
  )
fi

cd "$ROOT"
nohup "$ENV/bin/python" -m kareserve.lmcache_bridge \
  "${server_args[@]}" \
  >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"
echo "started lmcache-server pid=$!"
