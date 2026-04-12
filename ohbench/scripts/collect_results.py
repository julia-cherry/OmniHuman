import argparse
import csv
import json
import os
from typing import Any, Dict, Optional


def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_first_data_row_csv(path: str) -> Optional[Dict[str, str]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                return row
    return None


def _read_sq_overall_score(path: str) -> Optional[float]:
    row = _read_first_data_row_csv(path)
    if not row:
        return None
    v = row.get("overall_score")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _read_identity_metrics(path: str) -> Dict[str, Optional[float]]:
    """
    identity_consistency CSV rows: id_csim_single / id_csim_double with overall_mean
    (same header as compute_double.py). Legacy rows id_csim_single,val are also recognized.
    """
    out: Dict[str, Optional[float]] = {
        "IC(id_csim_single)": None,
        "IC*(id_csim_double)": None,
    }
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if key == "id_csim_single" and len(row) >= 2 and row[1] != "":
                try:
                    out["IC(id_csim_single)"] = float(row[1])
                except ValueError:
                    pass
            elif key == "id_csim_double" and len(row) >= 2 and row[1] != "":
                try:
                    out["IC*(id_csim_double)"] = float(row[1])
                except ValueError:
                    pass
    return out


def _extract_video_quality_avgs(vq: Any) -> Dict[str, float]:
    """
    video_quality/evaluate.py saves:
      {"imaging_quality": [avg, ...], "dynamic_degree": [avg, ...]}
    """
    out: Dict[str, float] = {}
    if not isinstance(vq, dict):
        return out
    for key, metric_name in (
        ("imaging_quality", "IQ(imaging_quality)"),
        ("dynamic_degree", "DD(dynamic_degree)"),
    ):
        v = vq.get(key)
        if isinstance(v, list) and v:
            try:
                out[metric_name] = float(v[0])
            except Exception:
                pass
    return out


def _scalar_av_metric(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and "overall" in v:
        try:
            return float(v["overall"])
        except Exception:
            return None
    return None


def _extract_person_person_means(pp: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(pp, list):
        return None

    keys = ["IN", "ES", "LR"]
    scores: Dict[str, list] = {k: [] for k in keys}
    for entry in pp:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        for k in keys:
            sub = result.get(k)
            if isinstance(sub, dict):
                v = sub.get("score")
                if isinstance(v, (int, float)):
                    scores[k].append(float(v))

    means: Dict[str, Any] = {}
    for k in keys:
        vals = scores[k]
        means[k] = (sum(vals) / len(vals)) if vals else None
    return means


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", required=True, help="Folder containing per-metric outputs")
    p.add_argument("--name", required=True, help="Run name prefix")
    p.add_argument(
        "--prune",
        action="store_true",
        help="Delete per-metric result files after writing the unified JSON",
    )
    args = p.parse_args()

    root = os.path.abspath(args.result_root)
    name = args.name

    vq_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "evaluators", "video_quality", "evaluation_results")
    )
    paths = {
        "identity_csv": os.path.join(root, f"{name}_identity.csv"),
        "sq_csv": os.path.join(root, f"{name}_sq.csv"),
        "sq_mean_csv": os.path.join(root, f"{name}_sq_mean.csv"),
        "audio_quality_json": os.path.join(root, f"{name}_audio_quality.json"),
        "av_alignment_json": os.path.join(root, f"{name}_av_alignment.json"),
        "person_person_json": os.path.join(root, f"{name}_person_person.json"),
        "video_types_csv": os.path.join(root, f"{name}_video_types.csv"),
        "video_quality_eval_json": os.path.join(vq_dir, f"{name}_eval_results.json"),
        "video_quality_full_info_json": os.path.join(vq_dir, f"{name}_full_info.json"),
    }

    metrics: Dict[str, Any] = {}

    # --- video_quality (README names) ---
    video_quality: Dict[str, Any] = {}
    vq = _read_json(paths["video_quality_eval_json"])
    if vq:
        video_quality.update(_extract_video_quality_avgs(vq))

    id_metrics = _read_identity_metrics(paths["identity_csv"])
    for k, v in id_metrics.items():
        if v is not None:
            video_quality[k] = v

    av = _read_json(paths["av_alignment_json"])
    if av:
        ib = _scalar_av_metric(av.get("ib_av"))
        clap = _scalar_av_metric(av.get("clap_score"))
        if ib is not None:
            video_quality["V-A(imagebind-av)"] = ib
        if clap is not None:
            video_quality["T-A(clap_score)"] = clap

    if video_quality:
        metrics["video_quality"] = video_quality

    # --- audio_quality ---
    aq = _read_json(paths["audio_quality_json"])
    if aq:
        block = aq.get("metrics")
        if isinstance(block, dict) and block:
            metrics["audio_quality"] = block

    # --- speech_quality ---
    sq = _read_sq_overall_score(paths["sq_mean_csv"])
    if sq is not None:
        metrics["speech_quality"] = {"SQ": sq}

    # --- person_person ---
    pp = _read_json(paths["person_person_json"])
    if pp:
        means = _extract_person_person_means(pp)
        if means:
            metrics["person_person"] = means

    out: Dict[str, Any] = {"name": name, "metrics": metrics}

    out_json = os.path.join(root, f"{name}_all.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Wrote:", out_json)

    if args.prune:
        keep = {os.path.abspath(out_json)}
        removed = []
        for pth in paths.values():
            if not pth:
                continue
            ap = os.path.abspath(pth)
            if ap in keep:
                continue
            if os.path.isfile(ap):
                try:
                    os.remove(ap)
                    removed.append(os.path.basename(ap))
                except Exception:
                    pass
        if removed:
            print("Pruned:", ", ".join(sorted(removed)))


if __name__ == "__main__":
    main()
