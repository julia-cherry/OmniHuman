#!/usr/bin/env bash
# =============================================================================
# person_person — Two-person video LLM judge: IN, ES, LR (API)
#
# Positional args:
#   $1  VIDEO_DIR    Videos (*.mp4 / *.mov)
#   $2  OUTPUT_JSON  Per-video JSON (includes raw model result)
#   $3  API_KEY      Provider API key (do not commit)
#   $4  MODEL        Optional; default gemini-2.5-pro; override with PERSON_PERSON_MODEL
#
# Optional env:
#   PERSON_PERSON_API_BASE  API root URL (if your provider requires a custom base URL)
#
# Example:
#   bash run.sh /data/videos /data/out/person.json "$YOUR_API_KEY"
# =============================================================================
set -euo pipefail

VIDEO_DIR="${1:?Missing $1: VIDEO_DIR}"
OUTPUT_JSON="${2:?Missing $2: OUTPUT_JSON}"
API_KEY="${3:?Missing $3: API_KEY}"
MODEL="${4:-${PERSON_PERSON_MODEL:-gemini-2.5-pro}}"
VIDEO_TYPE_CSV="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Do not set a default base URL here (avoid hardcoding provider endpoints).
# If your provider requires a custom base URL, set PERSON_PERSON_API_BASE or pass --api_base to run_all.sh.
if [[ -n "${PERSON_PERSON_API_BASE:-}" ]]; then
  export PERSON_PERSON_API_BASE
fi

echo "=== person_person (LLM) ==="
python3 evaluate_interaction.py \
  --video_dir "${VIDEO_DIR}" \
  --output_file "${OUTPUT_JSON}" \
  --api_key "${API_KEY}" \
  --model "${MODEL}" \
  ${VIDEO_TYPE_CSV:+--video_type_csv "${VIDEO_TYPE_CSV}"}
echo "=== person_person done ==="
