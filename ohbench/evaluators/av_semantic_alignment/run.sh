#!/usr/bin/env bash
# =============================================================================
# av_semantic_alignment — ImageBind AV + CLAP audio-text (ib_av, clap_score)
#
# Positional args:
#   $1  INFER_DIR    Generated samples: {basename}.mp4 and {basename}.wav per CSV rows
#   $2  INPUT_CSV    Table: path (for basename), text, optional audio_path / audio_text
#   $3  OUTPUT_JSON  Output JSON (ib_av, clap_score)
#   $4  DEVICE       Optional, default cuda:0 (first visible GPU when CUDA_VISIBLE_DEVICES set)
#
# Example:
#   bash run.sh /data/merged /data/prompts.csv /data/out/av.json cuda:0
# =============================================================================
set -euo pipefail

INFER_DIR="${1:?Missing $1: INFER_DIR}"
INPUT_CSV="${2:?Missing $2: INPUT_CSV}"
OUTPUT_JSON="${3:?Missing $3: OUTPUT_JSON}"
DEVICE="${4:-cuda:0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export AV_ALIGNMENT_WEIGHTS_DIR="${AV_ALIGNMENT_WEIGHTS_DIR:-${OH_ROOT}/models}"

python evaluate.py \
  --device "${DEVICE}" \
  --input_file "${INPUT_CSV}" \
  --infer_data_dir "${INFER_DIR}" \
  --output_file "${OUTPUT_JSON}" \
  --num_workers 0

echo "av_semantic_alignment done."
