#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zn/xyz/serve1
UV=${UV_BIN:-/home/zn/.local/bin/uv}
BASE_ENV=${BASE_VLLM_ENV:-"$ROOT/.venv-vllm-0.26"}
LMCACHE_ENV=${LMCACHE_ENV:-"$ROOT/.venv-vllm-0.26-lmcache"}
PYTHON_VERSION=python3.12

if [[ ! -x "$LMCACHE_ENV/bin/python" ]]; then
  "$UV" venv --python "$BASE_ENV/bin/python" "$LMCACHE_ENV"
fi

site_packages="$LMCACHE_ENV/lib/$PYTHON_VERSION/site-packages"
printf '%s\n' \
  "$BASE_ENV/lib/$PYTHON_VERSION/site-packages" \
  >"$site_packages/vllm_base_overlay.pth"

"$UV" pip install \
  --no-deps \
  --python "$LMCACHE_ENV/bin/python" \
  lmcache==0.5.2 \
  sortedcontainers==2.4.0 \
  cupy-cuda13x==14.1.1

"$LMCACHE_ENV/bin/python" -c \
  "import cupy, lmcache, torch, vllm; print(vllm.__version__, lmcache.__version__, torch.__version__, cupy.__version__)"
"$LMCACHE_ENV/bin/python" -c \
  "from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import LMCacheMPConnector; print(LMCacheMPConnector)"
