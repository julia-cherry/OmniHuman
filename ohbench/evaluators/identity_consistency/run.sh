#!/usr/bin/env bash
# =============================================================================
# identity_consistency — ArcFace ID similarity (single + two-person)
#
# Runs compute_single.py (single-speaker GT vs video) then compute_double.py
# (two-person GT vs video); both append rows to the same CSV (see Python scripts).
#
# Positional args:
#   $1  VIDEO_DIR   Generated videos (paired with GT by basename, e.g. foo.mp4 ↔ foo.jpg)
#   $2  OUTPUT_CSV  Output CSV (created if missing; double appends)
#   $3  GT_SINGLE   Single-person reference face images
#   $4  GT_DOUBLE   Two-person reference images (both faces in one image)
#
# GPU: uses CUDA_VISIBLE_DEVICES (run_all.sh / EVAL_CUDA_DEVICES).
#
# Example:
#   export CUDA_VISIBLE_DEVICES=0
#   bash run.sh /data/gen_videos /data/out/id.csv /data/gt_single /data/gt_double
# =============================================================================
set -euo pipefail

VIDEO_DIR="${1:?Missing $1: VIDEO_DIR (see header in run.sh)}"
OUTPUT="${2:?Missing $2: OUTPUT_CSV}"
GT_SINGLE="${3:?Missing $3: GT_SINGLE_DIR}"
GT_DOUBLE="${4:?Missing $4: GT_DOUBLE_DIR}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

python compute_single.py --gt_dir "${GT_SINGLE}" --video_dir "${VIDEO_DIR}" --output "${OUTPUT}"
python compute_double.py --gt_dir "${GT_DOUBLE}" --video_dir "${VIDEO_DIR}" --output "${OUTPUT}"

echo "identity_consistency done: ${OUTPUT}"
