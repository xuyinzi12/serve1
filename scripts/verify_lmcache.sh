#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
ENV="$ROOT/.venv-vllm-0.26"
GPU_ID=${GPU_IDS:-1}
VLLM_PORT=${KARESERVE_TEST_VLLM_PORT:-8102}

if [[ "$GPU_ID" != "1" ]]; then
  echo "LMCache verification uses GPU1 and port 8102" >&2
  exit 2
fi

cleanup() {
  bash "$ROOT/scripts/stop.sh" || true
}
trap cleanup EXIT

GPU_IDS=1 \
KARESERVE_ENABLE_LMCACHE=1 \
KARESERVE_CONFIG_PATH="$ROOT/configs/config.json" \
bash "$ROOT/scripts/start.sh"

"$ENV/bin/python" "$ROOT/scripts/lmcache_probe.py" \
  --base-url "http://127.0.0.1:$VLLM_PORT" \
  --phase store

for pid_file in "$ROOT/runtime"/pids/vllm-*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  kill "$pid"
  for _ in {1..30}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid"
  fi
  rm -f -- "$pid_file"
done

GPU_IDS=1 KARESERVE_ENABLE_LMCACHE=1 bash "$ROOT/scripts/start.sh" vllm

"$ENV/bin/python" "$ROOT/scripts/lmcache_probe.py" \
  --base-url "http://127.0.0.1:$VLLM_PORT" \
  --phase retrieve
