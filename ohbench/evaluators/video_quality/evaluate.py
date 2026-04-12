import argparse
import importlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import torch


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prompt_from_filename(name: str) -> str:
    prompt = Path(name).stem
    number_ending = r"-\d+$"
    if re.search(number_ending, prompt):
        return re.sub(number_ending, "", prompt)
    return prompt


def build_custom_image_dict(folder):
    d = {}
    for fn in os.listdir(folder):
        p = Path(folder) / fn
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            d[p.stem] = str(p)
    return d


def build_full_info_json(videos_path, output_path, name, dims, custom_image_folder):
    image_dict = build_custom_image_dict(custom_image_folder) if custom_image_folder else {}
    info = []
    for fn in os.listdir(videos_path):
        p = Path(videos_path) / fn
        if p.suffix.lower() not in {".mp4", ".gif"}:
            continue
        item = {
            "prompt_en": prompt_from_filename(fn),
            "dimension": dims,
            "video_list": [str(p)],
        }
        if p.stem in image_dict:
            item["custom_image_path"] = image_dict[p.stem]
        info.append(item)
    os.makedirs(output_path, exist_ok=True)
    out = str(Path(output_path) / f"{name}_full_info.json")
    save_json(info, out)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_path", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--dimension", nargs="+", required=True)
    p.add_argument("--custom_image_folder", default=None)
    p.add_argument("--output_path", default="./evaluation_results/")
    p.add_argument("--imaging_quality_preprocessing_mode", default="longer")
    return p.parse_args()


def main():
    args = parse_args()
    dims = args.dimension
    for dim in dims:
        if dim not in {"imaging_quality", "dynamic_degree"}:
            raise ValueError(f"Unsupported dimension: {dim}")

    run_name = args.name or f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    info_json = build_full_info_json(
        args.videos_path, args.output_path, run_name, dims, args.custom_image_folder
    )

    from core.utils import init_submodules

    submods = init_submodules(dims, local=False, read_frame=False)
    device = torch.device("cuda")
    kwargs = {"imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode}

    results = {}
    for dim in dims:
        module = importlib.import_module(f"core.{dim}")
        fn = getattr(module, f"compute_{dim}")
        results[dim] = fn(info_json, device, submods[dim], **kwargs)

    out_json = str(Path(args.output_path) / f"{run_name}_eval_results.json")
    save_json(results, out_json)
    print(f"video_quality results saved to {out_json}")


if __name__ == "__main__":
    main()
