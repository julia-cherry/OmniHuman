#!/usr/bin/env python3
import logging
import math
import os
import os.path as osp

import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms

from .models.multimodal_preprocessors import SimpleTokenizer

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10

FILE_DIR = osp.dirname(osp.abspath(__file__))
BPE_PATH = osp.join(FILE_DIR, "bpe", "bpe_simple_vocab_16e6.txt.gz")


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    fbank = fbank.transpose(0, 1)
    n_frames = fbank.size(1)
    p = target_length - n_frames
    if n_frames > 0 and abs(p) / n_frames > 0.2:
        logging.warning("Large gap between audio n_frames(%d) and target_length (%d)", n_frames, target_length)
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    return fbank.unsqueeze(0)


def get_clip_timepoints(clip_sampler, duration):
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    return torch.cat(tokens, dim=0)


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate): int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]
        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images, size=(height, width), mode="bilinear", align_corners=False
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size

    cropped = images[:, :, y_offset: y_offset + size, x_offset: x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Unsupported num_crops")

    def forward(self, videos):
        assert isinstance(videos, list)
        assert all([video.ndim == 4 for video in videos])
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


class NormalizeVideo:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        if not self.inplace:
            clip = clip.clone()
        mean = torch.as_tensor(self.mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(self.std, dtype=clip.dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return clip


def _open_video_with_sample_rate(video_path, sample_rate):
    # compatibility: some pytorchvideo builds expose sample_rate in from_path,
    # others only in EncodedVideoDecord constructor.
    import inspect
    sig = inspect.signature(EncodedVideo.from_path)
    if "sample_rate" in sig.parameters:
        return EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            sample_rate=sample_rate,
        )
    from pytorchvideo.data.encoded_video_decord import EncodedVideoDecord
    import io
    import pathlib
    vp = pathlib.Path(video_path)
    return EncodedVideoDecord(io.FileIO(vp, "rb"), str(vp), decode_audio=False, sample_rate=sample_rate)


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = _open_video_with_sample_rate(video_path, sample_rate)
        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0
            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)
        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)
