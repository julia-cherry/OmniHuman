#!/usr/bin/env bash
# =============================================================================
# video_quality — imaging_quality + dynamic_degree (see evaluate.py)
#
# Positional args:
#   $1  VIDEO_DIR            Directory of videos to score
#   $2  OUTPUT_NAME         Run name / prefix (writes under evaluation_results/)
#   $3  CUSTOM_IMAGE_FOLDER Reference images (per evaluate.py contract)
#
# GPU: uses CUDA_VISIBLE_DEVICES (run_all.sh sets from paths.env EVAL_CUDA_DEVICES).
#
# Example:
#   export CUDA_VISIBLE_DEVICES=0
#   bash run.sh /data/videos my_run /data/ref_images
# =============================================================================
set -euo pipefail

VIDEO_DIR="${1:?Missing $1: VIDEO_DIR}"
OUTPUT_NAME="${2:?Missing $2: OUTPUT_NAME}"
CUSTOM_IMAGE_FOLDER="${3:?Missing $3: CUSTOM_IMAGE_FOLDER}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${OH_ROOT}/models}"

python evaluate.py \
  --name "${OUTPUT_NAME}" \
  --dimension imaging_quality dynamic_degree \
  --videos_path "${VIDEO_DIR}" \
  --custom_image_folder "${CUSTOM_IMAGE_FOLDER}" \
  --output_path "${SCRIPT_DIR}/evaluation_results/"

echo "video_quality done."
