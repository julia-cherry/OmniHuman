#!/usr/bin/env bash
# =============================================================================
# audio_quality — FD, KL, AbS, WER, LSE-C (see METRICS)
#
# Positional args:
#   $1  INPUT_DIR      Generated samples: {id}.mp4 / {id}.wav paired with benchmark JSON
#   $2  BENCHMARK_DIR  Benchmark root (--flat: {id}.json, optional {id}.jpg / {id}.wav GT)
#   $3  OUTPUT_JSON    Aggregated JSON path
#   $4  CUDA_DEVICES  Optional; defaults to CUDA_VISIBLE_DEVICES (run_all uses EVAL_CUDA_DEVICES)
#
# Metrics passed to evaluate.py --metrics:
#   FD, KL, AbS, WER, LSE-C
#
# Example:
#   bash run.sh /data/merged /data/bench_ref /data/out/audio.json "0,1,2,3"
# =============================================================================
set -euo pipefail

INPUT_DIR="${1:?Missing $1: INPUT_DIR}"
BENCHMARK_DIR="${2:?Missing $2: BENCHMARK_DIR}"
OUTPUT_JSON="${3:?Missing $3: OUTPUT_JSON}"
CUDA_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-0}}"
METRICS="FD,KL,AbS,WER,LSE-C"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export MODELS_PATH="${MODELS_PATH:-${OH_ROOT}/models}"

echo "=== audio_quality ==="
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 evaluate.py \
    --input_dir "${INPUT_DIR}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --flat \
    --metrics "${METRICS}" \
    --output_json "${OUTPUT_JSON}"

echo "=== audio_quality done ==="
