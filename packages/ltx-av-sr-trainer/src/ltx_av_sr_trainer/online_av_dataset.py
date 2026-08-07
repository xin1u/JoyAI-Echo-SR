"""Online audio-video dataset for super-resolution training (no preprocessing).

Loads raw MP4 files, produces HQ at target resolution and LQ at lower
condition resolution via bicubic downscale (fast, CPU-friendly).

Video degradation is deferred to the training step (GPU-accelerated).

Returns per sample:
    hq_video: (C, F, H_hq, W_hq) in [-1, 1]  — original video at target res
    lq_video: (C, F, H_lq, W_lq) in [-1, 1]  — bicubic downscaled (no degradation yet)
    hq_audio: (T,) in [-1, 1] or None         — original audio waveform
    lq_audio: (T,) in [-1, 1] or None         — degraded audio waveform
    audio_sr: int                              — audio sample rate
    fps: float                                 — video frame rate
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_av_sr_trainer import logger
from ltx_av_sr_trainer.degradation.audio_degradation import apply_audio_degradation
from ltx_av_sr_trainer.raw_av_dataset import RawAVDataset


class OnlineAVDataset(Dataset):
    """Loads raw MP4 + produces HQ/LQ at different resolutions.

    HQ is loaded at full target resolution. LQ is bicubic downscaled from HQ.
    Video degradation is NOT applied here — it's done on GPU in the training step.

    Args:
        data_root: Directory containing {uuid}.mp4 + {uuid}_caption.json pairs.
        hq_width: HQ output video width (target/refiner output resolution).
        hq_height: HQ output video height.
        lq_width: LQ condition video width (base model output resolution).
        lq_height: LQ condition video height.
        target_frames: Number of frames (must satisfy frames % 8 == 1).
        audio_sample_rate: Target audio sample rate.
        with_audio: Whether to load and degrade audio.
        seed: Random seed for degradation reproducibility.
    """

    def __init__(
        self,
        data_root: str,
        hq_width: int = 1920,
        hq_height: int = 1152,
        lq_width: int = 1280,
        lq_height: int = 736,
        target_frames: int = 121,
        audio_sample_rate: int = 44100,
        with_audio: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._inner = RawAVDataset(
            data_root=data_root,
            target_width=hq_width,
            target_height=hq_height,
            target_frames=target_frames,
            audio_sample_rate=audio_sample_rate,
            caption_lang="en",
            require_audio=with_audio,
            seed=seed,
        )
        self.hq_width = hq_width
        self.hq_height = hq_height
        self.lq_width = lq_width
        self.lq_height = lq_height
        self.with_audio = with_audio
        self.audio_sample_rate = audio_sample_rate
        self._seed = seed

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict:
        sample = self._inner[index]

        hq_video = sample["video"]  # (C, F, H_hq, W_hq) in [-1, 1]
        fps = sample["fps"]

        # Bicubic downscale only (fast); degradation done on GPU later
        lq_video = self._downscale_video(hq_video)

        result: dict = {
            "hq_video": hq_video,
            "lq_video": lq_video,
            "fps": fps,
            "caption": sample.get("caption", ""),
            "hq_audio": None,
            "lq_audio": None,
            "audio_sr": self.audio_sample_rate,
        }

        if self.with_audio and "audio" in sample:
            hq_audio = sample["audio"]  # (C, T) — original stereo/mono
            audio_sr = int(sample.get("audio_sample_rate", self.audio_sample_rate))
            rng_audio = random.Random(self._seed + index + 10000)
            lq_audio = apply_audio_degradation(
                hq_audio if hq_audio.dim() == 2 else hq_audio.unsqueeze(0),
                rng=rng_audio, sample_rate=audio_sr,
            )
            if hq_audio.dim() == 1:
                lq_audio = lq_audio.squeeze(0)
            result["hq_audio"] = hq_audio
            result["lq_audio"] = lq_audio
            result["audio_sr"] = audio_sr

        return result

    def _downscale_video(self, hq_video: Tensor) -> Tensor:
        """Bicubic downscale HQ to LQ resolution (no degradation)."""
        c, t, h, w = hq_video.shape
        if h == self.lq_height and w == self.lq_width:
            return hq_video

        frames_2d = rearrange(hq_video.unsqueeze(0).float(), "b c t h w -> (b t) c h w")
        lq = F.interpolate(
            frames_2d, size=(self.lq_height, self.lq_width),
            mode="bicubic", align_corners=False,
        )
        lq = rearrange(lq, "(b t) c h w -> b c t h w", b=1, t=t)
        return lq.squeeze(0).clamp(-1, 1)
