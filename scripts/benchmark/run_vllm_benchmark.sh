#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
ENV=${VLLM_ENV:-"$ROOT/.venv-vllm-0.26"}
BASE_URL=${KARESERVE_BENCH_BASE_URL:-http://127.0.0.1:8090}
ENDPOINT=${KARESERVE_BENCH_ENDPOINT:-/v1/completions}
BACKEND=${KARESERVE_BENCH_BACKEND:-openai}
MODEL_NAME=${KARESERVE_MODEL_NAME:-kareserve-opt-1.3b}
MODEL_PATH=${KARESERVE_MODEL:-/home/zn/llm_models/opt-1.3b}
TOKENIZER=${KARESERVE_TOKENIZER:-"$MODEL_PATH"}
DATASET_NAME=${KARESERVE_DATASET_NAME:-prefix_repetition}
DATASET_PATH=${KARESERVE_DATASET_PATH:-}
NUM_PROMPTS=${KARESERVE_NUM_PROMPTS:-128}
REQUEST_RATE=${KARESERVE_REQUEST_RATE:-20}
MAX_CONCURRENCY=${KARESERVE_MAX_CONCURRENCY:-64}
SEED=${KARESERVE_SEED:-0}
RESULT_DIR=${KARESERVE_RESULT_DIR:-"$ROOT/runtime/benchmarks"}
BENCH_LABEL=${KARESERVE_BENCH_LABEL:-windowed-prefix}
RESULT_FILENAME=${KARESERVE_RESULT_FILENAME:-"${BENCH_LABEL}-${DATASET_NAME}.json"}

args=(
  bench serve
  --backend "$BACKEND"
  --base-url "$BASE_URL"
  --endpoint "$ENDPOINT"
  --model "$MODEL_NAME"
  --tokenizer "$TOKENIZER"
  --dataset-name "$DATASET_NAME"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --seed "$SEED"
  --save-result
  --result-dir "$RESULT_DIR"
  --result-filename "$RESULT_FILENAME"
)

case "$DATASET_NAME" in
  prefix_repetition)
    args+=(
      --prefix-repetition-prefix-len "${KARESERVE_PREFIX_LEN:-512}"
      --prefix-repetition-suffix-len "${KARESERVE_SUFFIX_LEN:-64}"
      --prefix-repetition-num-prefixes "${KARESERVE_NUM_PREFIXES:-8}"
      --prefix-repetition-output-len "${KARESERVE_OUTPUT_LEN:-16}"
    )
    ;;
  sharegpt|timed_trace|custom)
    if [[ -z "$DATASET_PATH" ]]; then
      echo "KARESERVE_DATASET_PATH is required for $DATASET_NAME" >&2
      exit 2
    fi
    args+=(--dataset-path "$DATASET_PATH")
    ;;
  random)
    args+=(
      --random-input-len "${KARESERVE_RANDOM_INPUT_LEN:-576}"
      --random-output-len "${KARESERVE_OUTPUT_LEN:-16}"
      --random-prefix-len "${KARESERVE_RANDOM_PREFIX_LEN:-512}"
    )
    ;;
  *)
    if [[ -n "$DATASET_PATH" ]]; then
      args+=(--dataset-path "$DATASET_PATH")
    fi
    ;;
esac

if [[ "${KARESERVE_BENCH_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "$ENV/bin/vllm" "${args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RESULT_DIR"
exec "$ENV/bin/vllm" "${args[@]}"
