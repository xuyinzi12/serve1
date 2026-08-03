#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
ENV="$ROOT/.venv-vllm-0.26"
MODEL=${KARESERVE_MODEL:-/home/zn/llm_models/opt-1.3b}
RESULT_DIR=${KARESERVE_RESULT_DIR:-"$ROOT/runtime/benchmarks"}

args=(
  bench serve
  --backend openai
  --base-url "${KARESERVE_BENCH_BASE_URL:-http://127.0.0.1:8090}"
  --endpoint /v1/completions
  --model "${KARESERVE_MODEL_NAME:-kareserve-opt-1.3b}"
  --tokenizer "$MODEL"
  --dataset-name prefix_repetition
  --num-prompts "${KARESERVE_NUM_PROMPTS:-128}"
  --request-rate "${KARESERVE_REQUEST_RATE:-20}"
  --max-concurrency "${KARESERVE_MAX_CONCURRENCY:-64}"
  --seed "${KARESERVE_SEED:-0}"
  --temperature 0
  --ignore-eos
  --prefix-repetition-prefix-len "${KARESERVE_PREFIX_LEN:-512}"
  --prefix-repetition-suffix-len "${KARESERVE_SUFFIX_LEN:-64}"
  --prefix-repetition-num-prefixes "${KARESERVE_NUM_PREFIXES:-8}"
  --prefix-repetition-output-len "${KARESERVE_OUTPUT_LEN:-16}"
  --save-result
  --result-dir "$RESULT_DIR"
  --result-filename "prefix-repetition.json"
)

mkdir -p "$RESULT_DIR"
exec "$ENV/bin/vllm" "${args[@]}"
