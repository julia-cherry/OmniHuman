"""Two-person video LLM judge: IN, ES, LR."""
import argparse
import base64
import csv
import json
import os
import time
from typing import Optional

from openai import OpenAI

METRIC_KEYS = ["IN", "ES", "LR"]

SYS_PROMPT = (
    "You are a strict but neutral video judge. You evaluate two-person dialogue / dual-speaker videos "
    "(typically two people visible). Your task is to decide only from evidence you actually see or hear—"
    "do not invent plot details. When you cannot clearly tell something (e.g. gaze direction, lip sync), "
    "prefer \"unknown\" or mid-range scores rather than guessing."
)

USER_PROMPT = """
You will judge a two-person conversational video. Produce output strictly as valid JSON following the template below. Do not add any keys not in the template. If a field does not apply or cannot be determined reliably, use \"unknown\" or an empty string/array as appropriate.

CRITICAL VISUAL FOCUS (must follow):
- Prioritize precise facial cues over background details.
- Pay special attention to: lips/mouth shapes, lip closure/opening timing, jaw motion, cheek movement, tongue visibility (if any), and whether lip motion matches the speaking role.
- Also attend to eye contact, gaze shifts, blinks, micro-expressions, head nods, smiles, and turn-taking gestures.
- Do NOT assume who is speaking based on subtitles/overlays; rely on the visual mouth activity and the audio (if audible) as much as possible.
- If you CANNOT clearly see gaze direction, head motion, or lip sync (e.g., due to low resolution, blur, or occlusion), explicitly say so in evidence (e.g., "gaze not clearly visible") and treat that aspect as \"unknown\" rather than guessing.
- Treat "gaze roughly oriented toward the partner" (e.g., looking in the general direction of the other person rather than at the exact eyeball) as partial evidence of interaction. Only treat interaction naturalness as very poor when both people almost never orient gaze/head toward each other and instead stare only at the camera or away from each other.
- Never claim that the listener's mouth moves with the speaker or that there is simultaneous speaking motion UNLESS you can clearly see multiple cycles of synchronized mouth opening/closing on both sides. Conversely, if you see the listener repeatedly forming full speech-like mouth shapes in sync with the speaker's timing (not just brief interjections), treat this as a strong artifact.

Task:
Score the video along 3 dimensions (each 1-10; 1 worst, 10 best) and give brief evidence-based reasons. Keep reasons concise but specific (mention observable cues).
- Use the full 1–10 range: if the behavior looks mostly natural with only minor issues, scores should typically be in the 6–8 range. Reserve 1–3 only for clear, strong artifacts or obviously implausible behavior. If there is some evidence of plausible behavior (e.g., occasional gaze toward the partner, some responsive nods), prefer mid scores (4–7) instead of extreme lows.

Dimension definitions & scoring rubric:
1) Interaction Naturalness (IN):
   - What to check: gaze alignment/eye contact, natural gaze shifts toward the partner, timely nonverbal responses (nod, smile, eyebrow movement), mutual attentiveness, natural turn-taking.
   - Low scores (1-3): blank stares, no responsive cues, gaze almost never oriented toward the partner (e.g., both keep looking only at the camera or away from each other), or clearly robotic/unrelated reactions. Only use this range when there is effectively **no** observable attempt at interaction.
   - Mid (4-7): some cues exist but timing is off, limited gaze variety, gaze only roughly points toward the partner or only occasionally checks the partner, responses feel delayed or generic. If there is at least one or a few clear gaze shifts toward the partner or weak but genuine-looking attempts at interaction, you should prefer this mid range (4–7) rather than 1–3.
   - High (8-10): consistent eye contact dynamics, believable micro-responses, timing matches conversational beats.

2) Emotion Synchrony (ES):
   - Step A: infer the speaker A emotion while A is speaking (happy/sad/angry/surprised/neutral/other/unknown).
   - Step B: infer listener B emotion during A's speaking segments.
   - Score based on whether B's emotion/expressions reasonably align (or plausibly contrast) with A and the conversational context.
   - Low (1-3): B expression is frozen, contradictory without reason, or mismatched across the whole clip.
   - Mid (4-7): partial alignment, but frequent drifting, low intensity, or inconsistent micro-expressions.
   - High (8-10): B reactions track A's emotion with believable nuance and timing (including appropriate contrast when warranted).

3) Listener Realism (LR):
   - Key check: Does the listener's mouth stay appropriately still/relaxed when listening, and only move in ways consistent with brief interjections or backchanneling (e.g., short "mm-hm" or nodding sounds), not full speech?
   - Pay extremely close attention to whether **both people appear to be speaking the same content at the same time** (same rhythm/timing of mouth opening and closing, as if they are reading or saying the same sentence together), even when their faces are relatively small in the frame. This is a strong artifact, typically caused by incorrect audio assignment, and should significantly reduce the score.
   - Detect artifacts: abnormal mouth jitter, random lip flapping, copied/mirrored mouth motion from the speaker, both mouths moving as if both are speaking the same content simultaneously with similar rhythm (audio assignment error), inconsistent lip timing, the listener clearly forming full syllables/words while the other is speaking, or visually strange, non-meaningful, inhuman-looking lip movements (even if they are not full speech-like shapes) at any point in the clip.
   - When artifacts are present, you MUST consider **how much of the clip** is affected: if these artifacts appear in a large portion of the video (e.g., many segments, or most of the speaking time), treat this as severe and give a low score. If they only appear very briefly in a small fraction of the clip, you may give a somewhat higher (but still reduced) score, and explicitly describe that the issue is brief/limited in duration.
   - Low (1-3): strong artifacts that affect a substantial part of the clip (listener mouth moves like the active speaker for multiple syllables/words, repeated jitter/unnatural motion across the clip, constant or recurrent non-meaningful mouth movement, or two mouths clearly speaking the same utterance at the same time for more than a brief moment). When you see such clearly non-human-like mouth behavior over a non-trivial portion of the video, you SHOULD give a low score rather than a high one.
   - Mid (4-7): artifacts exist but are limited in time or intensity (e.g., brief segments where the listener seems to move their mouth like the speaker, or short bursts of jitter), while the rest of the clip looks mostly acceptable.
   - High (8-10): listener mouth behavior looks natural throughout; mouth movement occurs only when appropriate (e.g., brief reactions) and there is no speaker-motion leakage, no duplicated-speech effect (two mouths speaking the same content in sync), and no unexplained jitter. If you are unsure whether there is a subtle artifact — **especially when faces are small in the frame or hard to see clearly** — you must NOT give a 9–10 or claim that the mouth is “completely still”; instead, prefer a mid score (5–7) and explicitly state that mouth motion could not be confidently verified.

JSON Template (use the key "IN" exactly for dimension 1):
{
  "IN": {
    "score": 1,
    "evidence": [""],
    "reason": ""
  },
  "ES": {
    "score": 1,
    "speaker_A_emotion": "unknown",
    "speaker_B_emotion": "unknown",
    "evidence": [""],
    "reason": ""
  },
  "LR": {
    "score": 1,
    "artifacts_detected": [""],
    "evidence": [""],
    "reason": ""
  }
}
""".strip()


def _clean_result_text(text: str):
    if not isinstance(text, str):
        return text

    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        return json.loads(s)
    except Exception:
        return s


class VideoEvaluator:
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str]):
        # base_url: optional provider endpoint override
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def encode_video(self, video_path: str) -> str:
        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:video/mp4;base64,{video_base64}"

    def evaluate_all_videos(
        self,
        folder_path: str,
        output_file: str,
        only_basenames=None,
        sleep_s: float = 2.0,
    ):
        video_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith((".mp4", ".mov"))
        )
        if only_basenames is not None:
            only_basenames = set(only_basenames)
            video_files = [f for f in video_files if os.path.splitext(f)[0] in only_basenames]
        results = []

        print(f"Found {len(video_files)} videos.")

        for idx, filename in enumerate(video_files):
            video_path = os.path.join(folder_path, filename)
            print(f"[{idx + 1}/{len(video_files)}] {filename}")

            try:
                video_url = self.encode_video(video_path)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_PROMPT},
                                {"type": "image_url", "image_url": {"url": video_url}},
                            ],
                        },
                    ],
                    temperature=0.1,
                )

                content = response.choices[0].message.content
                parsed = _clean_result_text(content)

                entry = {"video": filename, "result": parsed}
                results.append(entry)

                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                time.sleep(sleep_s)

            except Exception as e:
                print(f"Failed {filename}: {e}")

        print("person-person evaluation finished.")


def _collect_scores_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {k: [] for k in METRIC_KEYS}
    if not isinstance(data, list):
        return scores

    for entry in data:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        # Backward compatible: accept old keys if present.
        key_aliases = {
            "IN": ["IN"],
            "ES": ["ES", "emotion_sync"],
            "LR": ["LR", "listener_realism"],
        }
        for key in METRIC_KEYS:
            sub = None
            for k2 in key_aliases.get(key, [key]):
                if k2 in result:
                    sub = result.get(k2)
                    break
            if isinstance(sub, dict):
                val = sub.get("score")
                if isinstance(val, (int, float)):
                    scores[key].append(float(val))
    return scores


def _print_score_summary(path: str):
    print(f"\n====== Summary ({path}) ======")
    scores = _collect_scores_from_file(path)

    for k in METRIC_KEYS:
        print(f" - {k}: n={len(scores[k])}")

    means = {}
    print("\nMeans:")
    for k in METRIC_KEYS:
        vals = scores[k]
        if vals:
            m = sum(vals) / len(vals)
            means[k] = m
            print(f" - {k}: {m:.4f}")
        else:
            print(f" - {k}: (no valid scores)")

    if means:
        overall = sum(means.values()) / len(means)
        print(f"\nMean of three metrics: {overall:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--api_key", required=True)
    p.add_argument("--model", default=os.environ.get("PERSON_PERSON_MODEL", "gemini-2.5-pro"))
    p.add_argument(
        "--base_url",
        default=os.environ.get("PERSON_PERSON_API_BASE", ""),
    )
    p.add_argument("--sleep_s", type=float, default=2.0)
    p.add_argument(
        "--video_type_csv",
        default=None,
        help="CSV with columns basename,type (single/double). When provided, only type=double is evaluated.",
    )
    args = p.parse_args()

    only_basenames = None
    if args.video_type_csv:
        only_basenames = set()
        with open(args.video_type_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                if (row.get("type") or "").strip().lower() == "double":
                    b = (row.get("basename") or "").strip()
                    if b:
                        only_basenames.add(b)

    base_url = (args.base_url or "").strip().rstrip("/") or None
    ev = VideoEvaluator(args.api_key, args.model, base_url=base_url)
    ev.evaluate_all_videos(args.video_dir, args.output_file, only_basenames=only_basenames, sleep_s=args.sleep_s)
    _print_score_summary(args.output_file)


if __name__ == "__main__":
    main()
