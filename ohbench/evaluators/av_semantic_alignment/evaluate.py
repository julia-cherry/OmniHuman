import argparse
import json
import os
import os.path as osp
from glob import glob

import pandas as pd
import torch

from src.metrics import calc_imagebind_score, calc_clap_score


class Evaluator:
    def __init__(self, input_file, infer_data_dir, output_file, device="cuda:0", **kwargs):
        self.df = pd.read_csv(input_file)
        self.infer_data_dir = infer_data_dir
        self.output_file = output_file
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs

    def gather_pred_paths(self):
        pred_v, pred_a, valid = [], [], []
        for i in range(len(self.df)):
            gt_v = self.df["path"].iloc[i]
            gt_a = self.df.get("audio_path", self.df["path"]).iloc[i]
            v_name = osp.splitext(osp.basename(gt_v))[0] + ".mp4"
            a_name = osp.splitext(osp.basename(gt_a))[0] + ".wav"
            v_path = osp.join(self.infer_data_dir, v_name)
            a_path = osp.join(self.infer_data_dir, a_name)
            if osp.exists(v_path) and osp.exists(a_path):
                pred_v.append(v_path)
                pred_a.append(a_path)
                valid.append(i)
        if not valid:
            raise FileNotFoundError("No valid pred video/audio pairs found.")
        if len(valid) < len(self.df):
            self.df = self.df.iloc[valid].reset_index(drop=True)
        self.df["pred_video_path"] = pred_v
        self.df["pred_audio_path"] = pred_a

    def run(self):
        self.gather_pred_paths()
        prompts = self.df["text"].tolist()
        audio_prompts = self.df.get("audio_text", self.df["text"]).tolist()
        pred_video = self.df["pred_video_path"].tolist()
        pred_audio = self.df["pred_audio_path"].tolist()

        _, _, ib_av = calc_imagebind_score(
            pred_video, pred_audio, prompts, audio_prompts, device=self.device
        )
        clap_score = calc_clap_score(
            pred_audio,
            audio_prompts,
            device=self.device,
            num_workers=self.kwargs.get("num_workers", 0),
        )

        out = {"ib_av": ib_av, "clap_score": clap_score}
        os.makedirs(osp.dirname(self.output_file) or ".", exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"av_semantic_alignment results saved to {self.output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--infer_data_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    ev = Evaluator(**vars(args))
    ev.run()


if __name__ == "__main__":
    main()
