#!/usr/bin/env bash
# =============================================================================
# prepare_av_inputs — build a merged folder (copy mp4 + extract wav) for AV metrics.
#
# Positional args:
#   $1  VIDEO_DIR   Source folder of *.mp4
#   $2  MERGED_DIR  Output folder: same-basename .mp4 copies + 16 kHz mono .wav via ffmpeg
#
# Example:
#   bash prepare_av_inputs.sh /data/gen /data/gen_audio_video_merged
# =============================================================================
set -euo pipefail

VIDEO_DIR="${1:?Missing $1: VIDEO_DIR}"
MERGED_DIR="${2:?Missing $2: MERGED_DIR}"

mkdir -p "${MERGED_DIR}"

shopt -s nullglob
for video in "${VIDEO_DIR}"/*.mp4; do
  base="$(basename "${video}" .mp4)"
  wav="${MERGED_DIR}/${base}.wav"
  out_mp4="${MERGED_DIR}/${base}.mp4"

  if [[ ! -f "${out_mp4}" ]]; then
    cp "${video}" "${out_mp4}"
  fi

  if [[ ! -f "${wav}" ]]; then
    ffmpeg -y -i "${video}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${wav}" >/dev/null 2>&1 || true
  fi
done

echo "Prepared merged inputs at: ${MERGED_DIR}"
