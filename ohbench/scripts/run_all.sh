#!/usr/bin/env bash
# =============================================================================
# CLI: directory of .mp4 files to evaluate + root directory for all outputs.
#   GT / benchmark assets live in configs/paths.env.
#
#   bash scripts/run_all.sh <mp4_dir> <results_dir> [--name prefix] [--api_key KEY]
#
# Merged cache: prepare_av_inputs.sh always writes to <input_dir>_audio_video_merged
#
# paths.env (set paths on your machine):
#   CUSTOM_IMAGE_FOLDER       → reference images for video_quality
#   GT_SINGLE_DIR             → single-person GT face images for identity
#   GT_DOUBLE_DIR             → two-person GT images for identity
#   AV_INPUT_CSV              → CSV for AV alignment (columns path, text, …)
#   AUDIO_QUALITY_BENCHMARK_DIR → audio metrics benchmark (json / refs / GT wav, …)
# Optional: MODELS_ROOT or MODELS_PATH (default: ohbench/models — all weights in that one folder)
#
# GPU: EVAL_CUDA_DEVICES, EVAL_TORCH_DEVICE in paths.env
#
# --only: omit it to run everything (+ person_person if --api_key is set).
#         pass one category to run that stage only.
# =============================================================================
set -u -o pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${PROJ_ROOT}/evaluators"

INPUT_DIR=""
OUTPUT_DIR=""

## optional
OPT_MODEL_NAME=""
ONLY=""
API_KEY=""
API_BASE=""
# (no --merged_dir; merged path is fixed)

if [[ $# -ge 2 && "${1:0:1}" != "-" ]]; then
  INPUT_DIR="${1:?}"
  OUTPUT_DIR="${2:?}"
  shift 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir|--in)
      INPUT_DIR="${2:?}"
      shift 2
      ;;
    --output_dir|--out)
      OUTPUT_DIR="${2:?}"
      shift 2
      ;;
    --name|--model_name)
      OPT_MODEL_NAME="${2:?}"
      shift 2
      ;;
    --only)
      ONLY="${2:?}"
      shift 2
      ;;
    --api_key)
      API_KEY="${2:?}"
      shift 2
      ;;
    --api_base)
      API_BASE="${2:?}"
      shift 2
      ;;
    -h|--help)
      echo "bash scripts/run_all.sh <mp4_dir> <results_dir> [--name X] [--api_key K] [--api_base URL] [--only CATEGORY]"
      echo "  Omit --only to run all metrics; --only runs one category. Merged dir: <mp4_dir>_audio_video_merged"
      exit 0
      ;;
    *)
      echo "Unknown: $1 (try --help)"
      exit 1
      ;;
  esac
done

source "${PROJ_ROOT}/configs/paths.env"

# ---------------------------------------------------------------------------
# Robust runner: do not abort entire pipeline on one metric failure.
# Always try to collect results into one JSON at the end.
# ---------------------------------------------------------------------------
FAILED_STAGES=()
run_stage() {
  local name="$1"
  shift
  echo
  echo ">>> stage: ${name}"
  "$@"
  local ec=$?
  if [[ $ec -ne 0 ]]; then
    echo "!!! stage failed (${name}) exit_code=${ec} (continuing)"
    FAILED_STAGES+=("${name}")
  fi
  return 0
}

# All checkpoints live under one directory (default: ohbench/models).
_WEIGHT_ROOT="${MODELS_PATH:-${MODELS_ROOT:-${PROJ_ROOT}/models}}"
export MODELS_ROOT="${MODELS_ROOT:-${_WEIGHT_ROOT}}"
export MODELS_PATH="${MODELS_PATH:-${_WEIGHT_ROOT}}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${_WEIGHT_ROOT}}"
export DNSMOS_MODEL_DIR="${DNSMOS_MODEL_DIR:-${_WEIGHT_ROOT}}"
export AV_ALIGNMENT_WEIGHTS_DIR="${AV_ALIGNMENT_WEIGHTS_DIR:-${_WEIGHT_ROOT}}"

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: need input and output dirs (see --help)"
  exit 1
fi

VIDEO_DIR="${INPUT_DIR}"
AUDIO_VIDEO_MERGED_DIR="${VIDEO_DIR%/}_audio_video_merged"

MODEL_NAME="${OPT_MODEL_NAME:-${MODEL_NAME:-ohbench_eval}}"

export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
SQ_WORKERS_EFFECTIVE="${SQ_WORKERS:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"

RESULT_ROOT="${OUTPUT_DIR}"
[[ "${RESULT_ROOT}" == /* ]] || RESULT_ROOT="${PROJ_ROOT}/${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}"

if [[ -n "${ONLY}" && "${ONLY}" == "person_person" && -z "${API_KEY}" ]]; then
  echo "ERROR: person_person needs --api_key"
  exit 1
fi

should_run() { [[ -z "${ONLY}" || "${ONLY}" == "$1" ]]; }
need_merged() {
  should_run "video_quality" || should_run "audio_quality" || should_run "speech_quality"
}

if need_merged; then
  # Always (re)run merge for deterministic smoke tests and simpler UX.
  run_stage "prepare_av_inputs" bash "${PROJ_ROOT}/scripts/prepare_av_inputs.sh" "${VIDEO_DIR}" "${AUDIO_VIDEO_MERGED_DIR}"
fi

if should_run "video_quality"; then
  run_stage "video_quality" bash "${EVAL_DIR}/video_quality/run.sh" "${VIDEO_DIR}" "${MODEL_NAME}" "${CUSTOM_IMAGE_FOLDER}"
  run_stage "identity_consistency" bash "${EVAL_DIR}/identity_consistency/run.sh" \
    "${VIDEO_DIR}" "${RESULT_ROOT}/${MODEL_NAME}_identity.csv" "${GT_SINGLE_DIR}" "${GT_DOUBLE_DIR}"
  run_stage "av_semantic_alignment" bash "${EVAL_DIR}/av_semantic_alignment/run.sh" \
    "${AUDIO_VIDEO_MERGED_DIR}" "${AV_INPUT_CSV}" "${RESULT_ROOT}/${MODEL_NAME}_av_alignment.json" \
    "${EVAL_TORCH_DEVICE:-cuda:0}"
fi

if should_run "audio_quality"; then
  run_stage "audio_quality" bash "${EVAL_DIR}/audio_quality/run.sh" \
    "${AUDIO_VIDEO_MERGED_DIR}" "${AUDIO_QUALITY_BENCHMARK_DIR}" \
    "${RESULT_ROOT}/${MODEL_NAME}_audio_quality.json" "${CUDA_VISIBLE_DEVICES}"
fi

if should_run "speech_quality" || should_run "audio_quality"; then
  run_stage "speech_quality" bash "${EVAL_DIR}/speech_quality/run.sh" \
    "${AUDIO_VIDEO_MERGED_DIR}" "${RESULT_ROOT}/${MODEL_NAME}_sq.csv" \
    "${SQ_WORKERS_EFFECTIVE}" "${CUDA_VISIBLE_DEVICES}"
fi

if [[ -n "${API_KEY}" ]] && should_run "person_person"; then
  if [[ -n "${API_BASE}" ]]; then
    export PERSON_PERSON_API_BASE="${API_BASE}"
  fi
  : "${VIDEO_TYPE_CSV:?Missing VIDEO_TYPE_CSV in configs/paths.env}"
  [[ -f "${VIDEO_TYPE_CSV}" ]] || { echo "ERROR: VIDEO_TYPE_CSV not found: ${VIDEO_TYPE_CSV}"; exit 1; }
  echo "Using VIDEO_TYPE_CSV: ${VIDEO_TYPE_CSV}"
  run_stage "person_person" bash "${EVAL_DIR}/person-person/run.sh" \
    "${VIDEO_DIR}" "${RESULT_ROOT}/${MODEL_NAME}_person_person.json" "${API_KEY}" "" "${VIDEO_TYPE_CSV}"
fi

echo
echo "=== collect results ==="
python3 "${PROJ_ROOT}/scripts/collect_results.py" --result_root "${RESULT_ROOT}" --name "${MODEL_NAME}" --prune || true

if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
  echo
  echo "WARNING: some stages failed: ${FAILED_STAGES[*]}"
fi

echo "Done: ${RESULT_ROOT}"
