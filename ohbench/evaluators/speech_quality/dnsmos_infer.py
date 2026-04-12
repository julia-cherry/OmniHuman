"""DNSMOS (non-personalized) batch inference for SQ metrics. Models live in ./models/."""
import argparse
import concurrent.futures
import glob
import multiprocessing
import os
import sys

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class ComputeScore:
    def __init__(self, primary_model_path: str, p808_model_path: str) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.reshape(-1)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        aud = np.asarray(aud)
        if aud.ndim > 1:
            aud = np.mean(aud, axis=1)
        aud = aud.reshape(-1)
        if aud.size == 0:
            raise ValueError("empty audio")
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        audio = np.asarray(audio).reshape(-1).astype("float32")
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.concatenate([audio, audio])

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = (
                np.array(self.audio_melspec(audio=audio_seg[:-160])).astype("float32")[np.newaxis, :, :]
            )
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        return {
            "filename": fpath,
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
            "num_hops": num_hops,
            "OVRL_raw": float(np.mean(predicted_mos_ovr_seg_raw)),
            "SIG_raw": float(np.mean(predicted_mos_sig_seg_raw)),
            "BAK_raw": float(np.mean(predicted_mos_bak_seg_raw)),
            "OVRL": float(np.mean(predicted_mos_ovr_seg)),
            "overall_score": float(np.mean(predicted_mos_ovr_seg)),
            "SIG": float(np.mean(predicted_mos_sig_seg)),
            "BAK": float(np.mean(predicted_mos_bak_seg)),
            "P808_MOS": float(np.mean(predicted_p808_mos)),
        }


_compute_score = None


def _init_worker(counter, gpu_ids, primary_model_path, p808_model_path):
    global _compute_score
    with counter.get_lock():
        my_id = counter.value
        counter.value += 1
    gpu_id = gpu_ids[my_id % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _compute_score = ComputeScore(primary_model_path, p808_model_path)


def _run_one(args):
    clip, desired_fs = args
    return _compute_score(clip, desired_fs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset_dir", required=True)
    parser.add_argument("-o", "--csv_path", required=True)
    parser.add_argument(
        "--model_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Directory containing sig_bak_ovr.onnx and model_v8.onnx",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    primary_model_path = os.path.join(args.model_dir, "sig_bak_ovr.onnx")
    p808_model_path = os.path.join(args.model_dir, "model_v8.onnx")
    for p in (primary_model_path, p808_model_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing ONNX model: {p}")

    models = glob.glob(os.path.join(args.testset_dir, "*"))
    clips = list(glob.glob(os.path.join(args.testset_dir, "*.wav")))
    for m in tqdm(models, desc="Collecting clips"):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    clips = sorted(dict.fromkeys(clips))
    if not clips:
        print("ERROR: no .wav files under", args.testset_dir, file=sys.stderr)
        sys.exit(1)

    desired_fs = SAMPLING_RATE
    rows = []

    if args.workers <= 1 and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

    if args.workers <= 1:
        scorer = ComputeScore(primary_model_path, p808_model_path)
        for clip in tqdm(clips, desc="DNSMOS (SQ)", unit="clip"):
            try:
                rows.append(scorer(clip, desired_fs))
            except Exception as exc:
                print("%r: %s" % (clip, exc))
    else:
        counter = multiprocessing.Value("i", 0)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(counter, gpu_ids, primary_model_path, p808_model_path),
        ) as executor:
            future_to_clip = {
                executor.submit(_run_one, (clip, desired_fs)): clip for clip in clips
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_clip),
                total=len(future_to_clip),
                desc="DNSMOS (SQ)",
                unit="clip",
            ):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    print("%r: %s" % (future_to_clip[future], exc))

    df = pd.DataFrame(rows)
    if df.empty:
        print("ERROR: DNSMOS produced no scores (all clips failed?)", file=sys.stderr)
        sys.exit(1)
    df.to_csv(args.csv_path, index=False)
    detail_cols = ["SIG_raw", "BAK_raw", "OVRL", "SIG", "BAK", "P808_MOS"]
    exist_detail = [c for c in detail_cols if c in df.columns]
    if exist_detail:
        # For benchmark summary we only persist one number: overall_score.
        sq = float(df["OVRL"].mean()) if "OVRL" in df.columns else float("nan")
        mean_df = pd.DataFrame([{"overall_score": sq}])
        mean_path = args.csv_path.replace(".csv", "_mean.csv")
        mean_df.to_csv(mean_path, index=False)
        print("Mean metrics:", mean_path)


if __name__ == "__main__":
    main()
