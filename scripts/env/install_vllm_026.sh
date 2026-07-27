#!/usr/bin/env bash
set -euo pipefail

root=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
target_env="$root/.venv-vllm-0.26"
uv_bin=/home/zn/.local/bin/uv
index_url=${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

available_kb=$(df --output=avail "$root" | tail -n 1)
minimum_kb=$((35 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
    echo "available disk space is below 35 GiB" >&2
    exit 2
fi

if [[ ! -x "$target_env/bin/python" ]]; then
    "$uv_bin" venv --python 3.12 "$target_env"
fi

"$uv_bin" pip install \
    --python "$target_env/bin/python" \
    --index-url "$index_url" \
    vllm==0.26.0

"$target_env/bin/python" -c \
    'import torch, vllm; print("vllm", vllm.__version__); print("torch", torch.__version__); print("cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0))'
