#!/usr/bin/env bash
# =============================================================================
# speech_quality — DNSMOS (SQ), score all .wav in a directory
#
# Positional args:
#   $1  WAV_INPUT_DIR   Directory of wav files (often same as prepare_av_inputs merged dir)
#   $2  OUTPUT_CSV      Per-file CSV; *_mean.csv summary alongside
#   $3  WORKERS         Worker count; often equals GPU count
#   $4  GPU_IDS         Comma-separated physical GPU ids, round-robin with workers
#
# Example:
#   bash run.sh /data/merged /data/out/sq.csv 8 "0,1,2,3,4,5,6,7"
# =============================================================================
set -euo pipefail

INPUT_DIR="${1:?Missing $1: WAV_INPUT_DIR}"
OUTPUT_CSV="${2:?Missing $2: OUTPUT_CSV}"
WORKERS="${3:?Missing $3: WORKERS}"
CUDA_DEVICES="${4:?Missing $4: GPU_IDS (comma-separated)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

MODEL_DIR="${DNSMOS_MODEL_DIR:-${OH_ROOT}/models}"

echo "=== speech_quality (SQ / DNSMOS) ==="
python3 dnsmos_infer.py -t "${INPUT_DIR}" -o "${OUTPUT_CSV}" --workers "${WORKERS}" --gpus "${CUDA_DEVICES}" \
  --model_dir "${MODEL_DIR}"
echo "=== speech_quality done ==="
