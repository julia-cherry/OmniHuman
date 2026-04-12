import os
from fd.clap_inferencer import ClapInferencer
from audio_box.audio_box_inferencer import AudioBoxInferencer
from syncnet.syncnet_inferencer import SyncnetInferencer
from kl.kld_inferencer import KLDInferencer
from wer.wer_inferencer import WERInferencer
from tqdm import tqdm
from moviepy.editor import *
import json
import argparse

data_dir = "test_assets"

# ==============================================================================
# 1. Config: constants and metric bounds
# ==============================================================================

# Normalization range per metric [best, worst]
# --- main tuning knob for scoring ---
METRIC_BOUNDS = {
    'FD': {'best': 0.0, 'worst': 3.0},
    'KL': {'best': 0.0, 'worst': 4.0},
    'AbS': {'best': 7.25, 'worst': -1.75},  # (CE+CU+PQ-PC)/4, best=(10+10+10-1)/4, worst=(1+1+1-10)/4
    'WER': {'best': 0.0, 'worst': 1.0},
    'LSE-C': {'best': 10.0, 'worst': 0.0},
}

ALL_EVAL_METRICS = frozenset(METRIC_BOUNDS.keys())

# Metrics where higher raw values are better
HIGHER_IS_BETTER_METRICS = {'AbS', 'LSE-C'}

# Weights for sub-scores (audio vs sync/TTS); joint/video buckets removed with retired metrics.
WEIGHTS = {
    'audio': 0.2,
    'other': 0.1,
}


# ==============================================================================
# 2. Core scoring (usually unchanged)
# ==============================================================================

def normalize_metric(metric_name, value):
    """Normalize one metric using predefined bounds and direction."""
    if value is None:
        return None
    # -999 is used as a sentinel for "not computed / missing"
    try:
        if float(value) == -999:
            return None
    except Exception:
        return None

    bounds = METRIC_BOUNDS[metric_name]
    best, worst = bounds['best'], bounds['worst']

    if best == worst:
        return 0.5

    if metric_name in HIGHER_IS_BETTER_METRICS:
        norm_score = (value - worst) / (best - worst)
    else:
        norm_score = (worst - value) / (worst - best)

    return max(0, min(1, norm_score))


def calculate_overall_score(metrics):
    """Compute sub-scores and final overall score for one run."""
    normalized_scores = {name: normalize_metric(name, val) for name, val in metrics.items()}

    def _mean_available(keys):
        vals = []
        for k in keys:
            v = normalized_scores.get(k)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return sum(vals) / len(vals)

    s_audio = _mean_available(['FD', 'KL', 'AbS'])
    s_other = _mean_available(['WER', 'LSE-C'])

    parts = {
        'audio': s_audio,
        'other': s_other,
    }
    available_weight = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
    if available_weight <= 0:
        overall_score = None
    else:
        overall_score = sum((WEIGHTS[k] / available_weight) * float(v) for k, v in parts.items() if v is not None)

    return {
        "S_audio": s_audio,
        "S_other": s_other,
        "Overall Score": overall_score
    }


def evaluate_audiobox_wav(wav_path, audio_box_inferencer):
    score = audio_box_inferencer.infer(wav_path)
    return score['CE'], score['CU'], score['PC'], score['PQ']


def _iter_benchmark_items(sets_root, flat_mode):
    """Yield benchmark items: single dir if flat_mode else set1/set2/set3."""
    if flat_mode:
        for f in os.listdir(sets_root):
            if f.endswith(".json"):
                base_name = f[:-5]
                yield base_name, f"{sets_root}/{f}", sets_root
    else:
        for sez in ["set1", "set2", "set3"]:
            d = f"{sets_root}/{sez}"
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".json"):
                    base_name = f[:-5]
                    yield base_name, f"{d}/{f}", d


def calculate_metrics(input_dir, modes_path, sets_root, flat_mode=False, frame_batch_size: int = 16, skip_metrics=None):
    skip_metrics = set(skip_metrics or [])
    _ = frame_batch_size  # reserved for CLI compatibility

    if 'AbS' in skip_metrics:
        audio_box_inferencer = None
    else:
        print("Loading AudioBox model...")
        audio_box_inferencer = AudioBoxInferencer(modes_path)
    print("Loading Syncnet model...")
    syncnet_inferencer = SyncnetInferencer(modes_path)
    if 'WER' in skip_metrics:
        wer_inferencer = None
    else:
        print("Loading WER model...")
        try:
            wer_inferencer = WERInferencer(modes_path)
        except Exception as e:
            print(f"[warn] WER model failed to load; skipping WER: {e}")
            wer_inferencer = None
            skip_metrics.add('WER')
    print("Loading CLAP model...")
    clap_inferencer = ClapInferencer(modes_path)
    print("Loading KLD model...")
    kld_inferencer = KLDInferencer()

    total_fd = []
    total_kl = []
    total_ce = []
    total_cu = []
    total_pc = []
    total_pq = []
    total_wer = []
    total_lse_c = []

    items = list(_iter_benchmark_items(sets_root, flat_mode))
    n_video_ok = 0
    n_wav_ok = 0
    n_gt_ok = 0
    n_skipped = 0
    for base_name, json_path, bench_dir in tqdm(items, desc="audio eval", unit="item"):
        try:
            item = json.load(open(json_path))
            video_path = f"{input_dir}/{base_name}.mp4"
            wav_path = f"{input_dir}/{base_name}.wav"
            gt_wav = f"{bench_dir}/{base_name}.wav"

            has_audio_prompt = bool(item.get("audio_prompt") and item["audio_prompt"][0])
            has_speech_text = bool(item.get("speech_prompt", {}).get("text"))
            has_gt_audio = os.path.exists(gt_wav)
            if has_gt_audio:
                n_gt_ok += 1

            if os.path.exists(video_path):
                n_video_ok += 1
                if has_speech_text:
                    if 'LSE-C' not in skip_metrics:
                        sync_score = syncnet_inferencer.infer(video_path)[1]
                        if sync_score is not None:
                            total_lse_c.append(sync_score)

            if os.path.exists(wav_path):
                n_wav_ok += 1
                if audio_box_inferencer is not None:
                    ce, cu, pc, pq = evaluate_audiobox_wav(wav_path, audio_box_inferencer)
                    total_ce.append(ce)
                    total_cu.append(cu)
                    total_pc.append(pc)
                    total_pq.append(pq)
                if has_audio_prompt and has_gt_audio:
                    if 'FD' not in skip_metrics:
                        fd = clap_inferencer.infer_fd(wav_path, gt_wav)
                        total_fd.append(fd)
                    if 'KL' not in skip_metrics:
                        kl = kld_inferencer.infer(wav_path, gt_wav)
                        total_kl.append(kl)
                if has_speech_text:
                    if 'WER' not in skip_metrics and wer_inferencer is not None:
                        wer = wer_inferencer.infer_audio_text(wav_path, item["speech_prompt"]["text"])
                        total_wer.append(wer)
        except Exception as e:
            n_skipped += 1
            print(f"\n[skip] {base_name}: {e}")
            continue

    total = len(items)
    print(f"\n[diag] total={total} | skipped={n_skipped} | has_video={n_video_ok} | has_wav={n_wav_ok} | has_gt_wav={n_gt_ok}")
    if n_video_ok == 0:
        print("  -> No videos found: ensure {base_name}.mp4 exists under --input_dir and matches JSON basenames in benchmark_dir")
    if n_wav_ok == 0:
        print("  -> No wav files found: ensure {base_name}.wav exists under --input_dir")

    fd = (sum(total_fd) / len(total_fd) if len(total_fd) > 0 else -999) if ('FD' not in skip_metrics) else None
    kl = (sum(total_kl) / len(total_kl) if len(total_kl) > 0 else -999) if ('KL' not in skip_metrics) else None
    ce = sum(total_ce) / len(total_ce) if len(total_ce) > 0 else -999
    cu = sum(total_cu) / len(total_cu) if len(total_cu) > 0 else -999
    pc = sum(total_pc) / len(total_pc) if len(total_pc) > 0 else -999
    pq = sum(total_pq) / len(total_pq) if len(total_pq) > 0 else -999
    wer = (sum(total_wer) / len(total_wer) if len(total_wer) > 0 else -999) if ('WER' not in skip_metrics) else None
    lse_c = (sum(total_lse_c) / len(total_lse_c) if len(total_lse_c) > 0 else -999) if ('LSE-C' not in skip_metrics) else None
    if 'AbS' in skip_metrics:
        abs_score = None
    elif ce != -999 and cu != -999 and pc != -999 and pq != -999:
        abs_score = (ce + cu + pq - pc) / 4.0
    else:
        abs_score = -999
    scores_dict = {
        "FD": fd,
        "KL": kl,
        "AbS": abs_score,
        "WER": wer,
        "LSE-C": lse_c,
    }
    overall_score = calculate_overall_score(scores_dict)
    return scores_dict, overall_score, n_skipped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="mini_testset",
                        help="Dir with generated {id}.mp4 and {id}.wav")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark_data",
                        help="Benchmark root: flat mode has {id}.json and {id}.wav (GT for FD/KL); else set1/set2/set3 subdirs")
    parser.add_argument("--flat", action="store_true",
                        help="Single benchmark directory (no set1/set2/set3)")
    parser.add_argument("--output_json", type=str, default="audio_quality_results.json",
                        help="Output JSON path")
    parser.add_argument("--frame_batch_size", type=int, default=16,
                        help="Unused placeholder (kept for script compatibility)")
    parser.add_argument("--skip_metrics", type=str, default="",
                        help="Comma-separated metrics to skip (--metrics wins if set). Valid: FD,KL,AbS,WER,LSE-C")
    parser.add_argument("--metrics", type=str, default="",
                        help="Comma-separated metrics to run only, e.g. FD,KL,AbS,WER,LSE-C")
    if "MODELS_PATH" in os.environ:
        models_path = os.environ['MODELS_PATH']
    else:
        models_path = "models"
        os.environ['MODELS_PATH'] = models_path

    args = parser.parse_args()
    if args.metrics.strip():
        requested = {s.strip() for s in args.metrics.split(",") if s.strip()}
        unknown = requested - ALL_EVAL_METRICS
        if unknown:
            parser.error(f"Unknown --metrics: {sorted(unknown)}. Valid: {sorted(ALL_EVAL_METRICS)}")
        skip_metrics = list(ALL_EVAL_METRICS - requested)
    else:
        skip_metrics = [s.strip() for s in (args.skip_metrics or "").split(",") if s.strip()]
    scores_dict, overall_score, n_skipped = calculate_metrics(
        args.input_dir,
        models_path,
        args.benchmark_dir,
        flat_mode=args.flat,
        frame_batch_size=args.frame_batch_size,
        skip_metrics=skip_metrics,
    )
    print(scores_dict)
    print(overall_score)

    # Persist full results
    output = {
        "metrics": scores_dict,
        "overall": overall_score,
        "n_skipped": n_skipped,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {args.output_json}")
