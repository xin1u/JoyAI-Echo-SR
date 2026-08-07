"""Video degradation pipeline for AV super-resolution training.
Adapted from JoySR's Real-ESRGAN two-stage degradation pipeline.
Input: (B, C, T, H, W) in [-1, 1], Output: same shape degraded video.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange


MAX_TENSOR_INDEX = 2**31 - 1


def _build_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        kernel = torch.zeros((kernel_size, kernel_size), device=device, dtype=torch.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2 + 1e-9))
    kernel = kernel / (kernel.sum() + 1e-9)
    return kernel.to(dtype=dtype).view(1, 1, kernel_size, kernel_size)


def _apply_gaussian_blur(frames: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    padding = kernel_size // 2
    kernel = _build_gaussian_kernel(kernel_size, sigma, frames.device, frames.dtype)
    kernel = kernel.repeat(frames.shape[1], 1, 1, 1)
    b, c, h, w = frames.shape
    padded_numel = b * c * (h + 2 * padding) * (w + 2 * padding)
    if padded_numel > MAX_TENSOR_INDEX:
        max_b = max(1, MAX_TENSOR_INDEX // (c * (h + 2 * padding) * (w + 2 * padding)))
        outputs = []
        for i in range(0, b, max_b):
            chunk = frames[i : i + max_b]
            chunk = F.pad(chunk, (padding, padding, padding, padding), mode="reflect")
            outputs.append(F.conv2d(chunk, kernel, groups=c))
        return torch.cat(outputs, dim=0)
    frames = F.pad(frames, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(frames, kernel, groups=c)


def _jpeg_simulator(frames: torch.Tensor, quality: int) -> torch.Tensor:
    scale = max(0.125, quality / 100.0 * 0.5)
    orig_size = frames.shape[-2:]
    small = F.interpolate(frames, scale_factor=scale, mode="bicubic")
    return F.interpolate(small, size=orig_size, mode="bicubic")


def _apply_usm_sharpening(frames: torch.Tensor, weight: float = 0.5, threshold: float = 10 / 255.0) -> torch.Tensor:
    blur = _apply_gaussian_blur(frames, kernel_size=51, sigma=1.5)
    residual = frames - blur
    mask = residual.abs() > threshold
    sharpened = frames + weight * residual
    return torch.where(mask, sharpened, frames).clamp(0.0, 1.0)


def _smart_resize(frames: torch.Tensor, scale: float, rng: random.Random) -> torch.Tensor:
    mode = rng.choice(["bilinear", "bicubic", "area"])
    h_new = max(16, int(frames.shape[-2] * scale))
    w_new = max(16, int(frames.shape[-1] * scale))
    h_new = h_new if h_new % 2 == 0 else h_new + 1
    w_new = w_new if w_new % 2 == 0 else w_new + 1
    return F.interpolate(frames, size=(h_new, w_new), mode=mode, align_corners=False if mode != "area" else None)


def _add_random_noise(
    frames: torch.Tensor,
    sigma_range: tuple[float, float],
    rng: random.Random,
    generator: torch.Generator,
) -> torch.Tensor:
    sigma = rng.uniform(*sigma_range)
    if rng.random() < 0.4:
        noise = (
            torch.randn(
                (frames.shape[0], 1, frames.shape[2], frames.shape[3]),
                dtype=frames.dtype,
                device=frames.device,
                generator=generator,
            )
            * sigma
        )
    else:
        noise = torch.randn(frames.shape, dtype=frames.dtype, device=frames.device, generator=generator) * sigma
    return (frames + noise).clamp(0.0, 1.0)


@torch.no_grad()
def _cutmix_fixed_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    num_frames: int,
    ratio_a: float = 0.4,
) -> torch.Tensor:
    n, c, h, w = a.shape
    out = b.clone()
    cut_rat = math.sqrt(ratio_a)
    cut_w = max(1, int(round(w * cut_rat)))
    cut_h = max(1, int(round(h * cut_rat)))

    for bi in range(batch_size):
        cx = int(torch.randint(0, w, (1,), device=a.device).item())
        cy = int(torch.randint(0, h, (1,), device=a.device).item())
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, x1 + cut_w)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, y1 + cut_h)
        x1 = max(0, x2 - cut_w)
        y1 = max(0, y2 - cut_h)
        sl = slice(bi * num_frames, (bi + 1) * num_frames)
        out[sl, :, y1:y2, x1:x2] = a[sl, :, y1:y2, x1:x2]

    return out


def _stage_one(frames: torch.Tensor, rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    frames = _apply_usm_sharpening(frames)
    k1 = rng.choice([21, 23, 25, 27, 31])
    s1 = rng.uniform(0.5, 5.0)
    frames = _apply_gaussian_blur(frames, k1, s1)
    frames = _smart_resize(frames, rng.uniform(0.3, 1.0), rng)
    frames = _add_random_noise(frames, (2 / 255.0, 25 / 255.0), rng, generator)
    if rng.random() < 0.9:
        frames = _jpeg_simulator(frames, rng.randint(50, 85))
    return frames


def _stage_two(frames: torch.Tensor, rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    frames = _apply_usm_sharpening(frames)
    k1 = rng.choice([15, 21, 23, 25, 27, 31, 37])
    s1 = rng.uniform(0.5, 5.0)
    frames = _apply_gaussian_blur(frames, k1, s1)
    frames = _smart_resize(frames, rng.uniform(0.1, 1.2), rng)
    frames = _add_random_noise(frames, (2 / 255.0, 25 / 255.0), rng, generator)
    if rng.random() < 0.9:
        frames = _jpeg_simulator(frames, rng.randint(40, 85))

    if rng.random() < 0.8:
        k2 = rng.choice([15, 21, 23, 25, 27, 31, 37])
        s2 = rng.uniform(0.3, 2.5)
        frames = _apply_gaussian_blur(frames, k2, s2)
    frames = _smart_resize(frames, rng.uniform(0.15, 1.0), rng)
    frames = _add_random_noise(frames, (2 / 255.0, 20 / 255.0), rng, generator)
    if rng.random() < 0.8:
        frames = _jpeg_simulator(frames, rng.randint(40, 85))
    return frames


@torch.no_grad()
def apply_video_degradation(
    video: torch.Tensor,
    rng: random.Random,
    generator: torch.Generator | None = None,
    downsample_factor: int = 2,
) -> torch.Tensor:
    """Apply two-stage Real-ESRGAN style degradation to video.
    Args:
        video: (B, C, T, H, W) in [-1, 1]
        rng: random state for reproducibility
        generator: torch generator for noise
        downsample_factor: spatial downsample then upsample back
    Returns:
        Degraded video (B, C, T, H, W) in [-1, 1], same spatial size as input
    """
    if generator is None:
        generator = torch.Generator(device=video.device).manual_seed(rng.randint(0, 2**31 - 1))

    b, c, t, h, w = video.shape
    dtype = video.dtype

    frames = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
    frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0).to(torch.float32)

    rand_s = float(rng.random())
    ds = downsample_factor if rand_s >= 0.1 else downsample_factor * 2

    branch = float(rng.random())
    if branch < 0.7:
        frames_out = _stage_two(frames, rng, generator)
        frames_out = F.interpolate(frames_out, size=(h // ds, w // ds), mode="bicubic", align_corners=False)
        frames_out = F.interpolate(frames_out, size=(h, w), mode="bicubic", align_corners=False)
    elif branch < 0.75:
        frames_out = _stage_one(frames, rng, generator)
        frames_out = F.interpolate(frames_out, size=(h // ds, w // ds), mode="bicubic", align_corners=False)
        frames_out = F.interpolate(frames_out, size=(h, w), mode="bicubic", align_corners=False)
    else:
        frames_real = _stage_two(frames, rng, generator)
        frames_light = _stage_one(frames, rng, generator)
        aigc_ds = downsample_factor if ds == downsample_factor * 2 and rng.random() < 0.8 else ds
        frames_light = F.interpolate(frames_light, size=(h // aigc_ds, w // aigc_ds), mode="bicubic", align_corners=False)
        frames_light = F.interpolate(frames_light, size=(h, w), mode="bicubic", align_corners=False)
        frames_real = F.interpolate(frames_real, size=(h // ds, w // ds), mode="bicubic", align_corners=False)
        frames_real = F.interpolate(frames_real, size=(h, w), mode="bicubic", align_corners=False)
        ratio_a = rng.uniform(0.3, 0.4)
        frames_out = _cutmix_fixed_ratio(frames_light, frames_real, b, t, ratio_a=ratio_a)

    frames_out = (frames_out * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype)
    return rearrange(frames_out, "(b t) c h w -> b c t h w", b=b, t=t)
