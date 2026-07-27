#!/usr/bin/env bash
set -u

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

check_index() {
    local name=$1
    local url=$2
    local metrics

    metrics=$(curl \
        --location \
        --silent \
        --show-error \
        --max-time 20 \
        --output "$tmp_file" \
        --write-out '%{http_code} %{time_total} %{size_download}' \
        "$url")
    local status=$?

    if (( status != 0 )); then
        printf '%s curl_status=%d\n' "$name" "$status"
        return
    fi

    if grep --quiet 'vllm-0.26.0' "$tmp_file"; then
        printf '%s http_time_bytes=%s version_0.26.0=yes\n' "$name" "$metrics"
    else
        printf '%s http_time_bytes=%s version_0.26.0=no\n' "$name" "$metrics"
    fi
}

check_index official https://pypi.org/simple/vllm/
check_index tuna https://pypi.tuna.tsinghua.edu.cn/simple/vllm/
check_index aliyun https://mirrors.aliyun.com/pypi/simple/vllm/
