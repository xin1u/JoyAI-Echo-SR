from __future__ import annotations

import random
from dataclasses import fields
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from .aigc_deg import DegradeConfig, SyntheticDegradation
from .degradation_utils import (
    _add_random_noise,
    _apply_gaussian_blur,
    _apply_sinc_filter,
    _jpeg_simulator,
    _smart_resize,
    _apply_usm_sharpening,
    cutmix_fixed_ratio,
)
from .ffmpeg_deg_utils import FFmpegCompressionDegradeBTCHW

__all__ = ["apply_aigc_realbasicvsr_degradation"]


def _build_aigc_config(config_dict: dict[str, Any] | None) -> DegradeConfig:
    if config_dict is None:
        return DegradeConfig()
    valid = {f.name for f in fields(DegradeConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid}
    return DegradeConfig(**filtered)


def _stage_one_realbasic(frames: torch.Tensor, rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    """Keep the second stage of Real-ESRGAN degradation."""
        # 第一步: USM 锐化
    frames = _apply_usm_sharpening(frames)

    # --- 第一阶退化 ---
    # 1. 模糊
    k1 = rng.choice([21, 23, 25, 27])
    s1 = rng.uniform(0.25, 3.0)
    frames = _apply_gaussian_blur(frames, k1, s1)
    # 2. 缩放
    frames = _smart_resize(frames, rng.uniform(0.25, 1.2), rng)
    # 3. 噪声
    frames = _add_random_noise(frames, (1/255., 15/255.), rng, generator)
    # 4. JPEG
    if rng.random() < 0.9:
        frames = _jpeg_simulator(frames, rng.randint(25, 95))

    return frames



def _stage_two_realbasic(frames: torch.Tensor, rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    """Keep the second stage of Real-ESRGAN degradation."""
        # 第一步: USM 锐化
    frames = _apply_usm_sharpening(frames)

    # --- 第一阶退化 ---
    # 1. 模糊
    k1 = rng.choice([15, 21, 23, 25, 27, 31, 37])
    s1 = rng.uniform(0.25, 3.0)
    frames = _apply_gaussian_blur(frames, k1, s1)
    # 2. 缩放
    frames = _smart_resize(frames, rng.uniform(0.25, 1.5), rng) # [0.15, 1.5]
    # 3. 噪声
    frames = _add_random_noise(frames, (1/255., 15/255.), rng, generator)
    # 4. JPEG
    if rng.random() < 0.9:
        frames = _jpeg_simulator(frames, rng.randint(70, 95))

    # --- 第二阶退化 ---
    # 1. 模糊 (80% 概率)
    if rng.random() < 0.8:
        k2 = rng.choice([15, 21, 23, 25, 27, 31, 37])
        s2 = rng.uniform(0.2, 1.5)
        frames = _apply_gaussian_blur(frames, k2, s2)
    # 2. 缩放
    frames = _smart_resize(frames, rng.uniform(0.25, 1.2), rng)
    # 3. 噪声
    frames = _add_random_noise(frames, (1/255., 15/255.), rng, generator)
    # 4. JPEG 或 Sinc (Shuffle 阶段)
    if rng.random() < 0.8:
        if rng.random() < 0.5:
            frames = _jpeg_simulator(frames, rng.randint(25, 95))
        else:
            # frames = _apply_sinc_filter(frames, kernel_size=17, rng=rng)
            frames = _jpeg_simulator(frames, rng.randint(70, 95))

    # # Optional blur5
    # if rng.random() < 0.8:
    #     # k2 = rng.choice([ 5, 7, 9, 11])
    #     k2 = rng.choice([ 11, 13, 15])
    #     # k2 = rng.choice([ 11, 15, 19, 21])
    #     s2 = rng.uniform(0.25, 1.5)
    #     frames = _apply_gaussian_blur(frames, k2, s2)
    # # Resize
    # # frames = _smart_resize(frames, rng.uniform(0.5, 1.2), rng)
    # # Noise
    # if rng.random() < 0.8:
    #     # frames = _add_random_noise(frames, (1 / 255.0, 5 / 255.0), rng, generator)
    #     frames = _add_random_noise(frames, (1 / 255.0, 20 / 255.0), rng, generator)
    # # JPEG or sinc
    # if rng.random() < 0.8:
    #     if rng.random() < 0.5:
    #         # frames = _jpeg_simulator(frames, rng.randint(70, 95))
    #         frames = _jpeg_simulator(frames, rng.randint(55, 85))
    #         frames = _apply_sinc_filter(frames, kernel_size=17, rng=rng)
    #     else:
    #         frames = _apply_sinc_filter(frames, kernel_size=17, rng=rng)
    #         # frames = _apply_sinc_filter(frames, kernel_size=13, rng=rng)
    return frames


def apply_aigc_realbasicvsr_degradation(
    video: torch.Tensor,
    rng: random.Random,
    generator: torch.Generator | None = None,
    aigc_config: dict[str, Any] | None = None,
    is_aigc: bool = True,
) -> torch.Tensor:
    """
    Hybrid degradation: stage-1 uses the SyntheticDegradation pipeline from AIGC_deg,
    stage-2 keeps the Real-ESRGAN operations.
    """
    if generator is None:
        generator = torch.Generator(device=video.device).manual_seed(rng.randint(0, 2**31 - 1))

    b, c, t, h, w = video.shape
    dtype = video.dtype
    device = video.device

    # Prepare data for AIGC degradation (B,T,C,H,W) in [0,1]
    frames_bt = video.permute(0, 2, 1, 3, 4).contiguous()
    frames_bt = ((frames_bt + 1.0) / 2.0).clamp(0.0, 1.0).to(torch.float32)

    ##################################### Apply AIGC Degradation
    # Stage-1: SyntheticDegradation
    # if is_aigc:
    #     degrader = SyntheticDegradation(_build_aigc_config(aigc_config)).to(device).eval()
    #     with torch.no_grad():
    #         frames_bt = degrader(frames_bt)


    ##################################### Apply FFmpeg Degradation

    ffmpeg_degrade = FFmpegCompressionDegradeBTCHW(
        crf_range=(25,30),
        prob=1.0,
    )
    frames_bt = ffmpeg_degrade(frames_bt)

    # Back to (B,C,T,H,W) then flatten for stage-2

    #############################################################
    frames = frames_bt.permute(0, 2, 1, 3, 4).contiguous()
    frames = rearrange(frames, "b c t h w -> (b t) c h w").contiguous()

    rand_s = float(rng.random())

    if rand_s < 0.1:
        ds_factor = 2
    else:
        ds_factor = 4

    # Stage-2: Real-ESRGAN second degradation stage
    random_num = float(rng.random())

    if  random_num< 0.7:
        frames_return = _stage_two_realbasic(frames, rng, generator)

        frames_return = F.interpolate(frames_return, size=(h // ds_factor, w // ds_factor), mode="bicubic", align_corners=False)
        frames_return = F.interpolate(frames_return, size=(h, w), mode="bicubic", align_corners=False)


    elif random_num >= 0.7 and random_num < 0.75:

        frames_return = _stage_one_realbasic(frames, rng, generator)
    # Force overall 2x spatial downsampling (matching original training recipe)
        frames_return = F.interpolate(frames_return, size=(h // ds_factor, w // ds_factor), mode="bicubic", align_corners=False)
        frames_return = F.interpolate(frames_return, size=(h, w), mode="bicubic", align_corners=False)

    else:
        frames_realesrdeg = _stage_two_realbasic(frames, rng, generator)
        frames_aigcdeg = _stage_one_realbasic(frames, rng, generator)
    # Force overall 2x spatial downsampling (matching original training recipe)
        aigc_ds_factor = ds_factor
        rand_mix = float(rng.random())
        if ds_factor == 4:
            if rand_mix < 0.8:
                aigc_ds_factor = 2

        frames_aigcdeg = F.interpolate(frames_aigcdeg, size=(h // aigc_ds_factor, w // aigc_ds_factor), mode="bicubic", align_corners=False)
        frames_aigcdeg = F.interpolate(frames_aigcdeg, size=(h, w), mode="bicubic", align_corners=False)

        frames_realesrdeg = F.interpolate(frames_realesrdeg, size=(h // ds_factor, w // ds_factor), mode="bicubic", align_corners=False)
        frames_realesrdeg = F.interpolate(frames_realesrdeg, size=(h, w), mode="bicubic", align_corners=False)
        # torch.
        ratio_a = random.uniform(0.3,0.4)
        # inpainting
        frames_return = cutmix_fixed_ratio(
            frames_aigcdeg, frames_realesrdeg,
            Bsz=b, T=t,
            ratio_a=ratio_a,
            p=1.0)

    frames_return = (frames_return * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype)


    return rearrange(frames_return, "(b t) c h w -> b c t h w", b=b, t=t)
