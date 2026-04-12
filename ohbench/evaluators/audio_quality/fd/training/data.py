"""
Minimal subset of LAION CLAP training/data.py for inference only (fd.hook).
"""
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision


def int16_to_float32(x):
    return (x / 32767.0).astype("float32")


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")


def get_mel(audio_data, audio_cfg):
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["window_size"],
        win_length=audio_cfg["window_size"],
        hop_length=audio_cfg["hop_size"],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg["mel_bins"],
        f_min=audio_cfg["fmin"],
        f_max=audio_cfg["fmax"],
    ).to(audio_data.device)

    mel = mel_tf(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T


def get_audio_features(
    sample,
    audio_data,
    max_len,
    data_truncating,
    data_filling,
    audio_cfg,
    require_grad=False,
):
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                mel = get_mel(audio_data, audio_cfg)
                chunk_frames = max_len // audio_cfg["hop_size"] + 1
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(
                        list(range(0, total_frames - chunk_frames + 1)), 3
                    )
                    if len(ranges[1]) == 0:
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        ranges[2] = [0]
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]
                    mel_shrink = torchvision.transforms.Resize(
                        size=[chunk_frames, audio_cfg["mel_bins"]]
                    )(mel[None])[0]
                    mel_fusion = torch.stack(
                        [mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back],
                        dim=0,
                    )
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx : idx + max_len]
        else:
            if len(audio_data) < max_len:
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == "fusion":
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data
    return sample
