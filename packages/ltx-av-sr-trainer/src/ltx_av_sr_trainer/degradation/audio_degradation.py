"""Audio degradation pipeline for AV super-resolution training.
Simulates old movie audio artifacts: bandwidth limitation, noise, clipping,
reverberation, and lossy compression artifacts.
Input: waveform (B, T) or (B, 1, T) at original sample rate.
Output: degraded waveform, same shape.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def _apply_lowpass(waveform: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    """Simple FIR lowpass via sinc kernel to simulate bandwidth limitation."""
    if cutoff_ratio >= 1.0:
        return waveform
    kernel_size = 101
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=waveform.device)
    omega = cutoff_ratio * torch.pi
    sinc = torch.where(t == 0, torch.ones_like(t) * omega / torch.pi, torch.sin(omega * t) / (torch.pi * t))
    window = 0.54 + 0.46 * torch.cos(torch.pi * t / half)
    kernel = sinc * window
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    needs_squeeze = waveform.dim() == 2
    if needs_squeeze:
        waveform = waveform.unsqueeze(1)

    padded = F.pad(waveform, (half, half), mode="reflect")
    out = F.conv1d(padded, kernel.to(waveform.dtype))
    if needs_squeeze:
        out = out.squeeze(1)
    return out


@torch.no_grad()
def _apply_highpass(waveform: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    """Simple FIR highpass to simulate telephone-band filtering."""
    if cutoff_ratio <= 0.0:
        return waveform
    return waveform - _apply_lowpass(waveform, cutoff_ratio)


@torch.no_grad()
def _add_noise(waveform: torch.Tensor, snr_db: float, rng: random.Random) -> torch.Tensor:
    """Add Gaussian noise at a given SNR."""
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10) + 1e-8)
    noise_std = noise_power.sqrt()
    generator = torch.Generator(device=waveform.device).manual_seed(rng.randint(0, 2**31 - 1))
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


@torch.no_grad()
def _add_hum(waveform: torch.Tensor, freq_hz: float, sample_rate: int, amplitude: float) -> torch.Tensor:
    """Add power-line hum (50Hz or 60Hz)."""
    length = waveform.shape[-1]
    t = torch.arange(length, dtype=torch.float32, device=waveform.device) / sample_rate
    if waveform.dim() == 3:
        t = t.unsqueeze(0).unsqueeze(0)
    else:
        t = t.unsqueeze(0)
    hum = amplitude * torch.sin(2 * torch.pi * freq_hz * t)
    return waveform + hum.to(waveform.dtype)


@torch.no_grad()
def _apply_clipping(waveform: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft clipping to simulate analog saturation."""
    return torch.tanh(waveform / (threshold + 1e-8)) * threshold


@torch.no_grad()
def _apply_mp3_artifact(waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Simulate lossy compression via aggressive low-pass + quantization noise."""
    cutoff = rng.uniform(0.3, 0.7)
    waveform = _apply_lowpass(waveform, cutoff)
    bits = rng.choice([8, 10, 12])
    levels = 2**bits
    waveform = torch.round(waveform * levels) / levels
    return waveform


@torch.no_grad()
def _apply_reverb(waveform: torch.Tensor, rng: random.Random, sample_rate: int) -> torch.Tensor:
    """Simple synthetic reverb via decaying delay taps."""
    delays_ms = [rng.uniform(10, 50) for _ in range(rng.randint(3, 6))]
    decays = [rng.uniform(0.1, 0.4) for _ in delays_ms]
    out = waveform.clone()
    for delay_ms, decay in zip(delays_ms, decays):
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples >= waveform.shape[-1]:
            continue
        delayed = F.pad(waveform[..., :-delay_samples], (delay_samples, 0))
        out = out + decay * delayed
    peak = out.abs().max()
    if peak > 0:
        out = out * (waveform.abs().max() / peak)
    return out


@torch.no_grad()
def apply_audio_degradation(
    waveform: torch.Tensor,
    rng: random.Random,
    sample_rate: int = 44100,
) -> torch.Tensor:
    """Apply random old-movie-style audio degradation.
    Args:
        waveform: (B, T) or (B, 1, T) audio tensor in [-1, 1]
        rng: random state
        sample_rate: audio sample rate
    Returns:
        Degraded waveform, same shape, in [-1, 1]
    """
    out = waveform.clone().float()

    # 1. Bandwidth limitation (95% probability, heavier)
    if rng.random() < 0.95:
        cutoff = rng.uniform(0.15, 0.75)
        out = _apply_lowpass(out, cutoff)

    # 2. High-pass (remove sub-bass rumble, 60% probability)
    if rng.random() < 0.6:
        hp_cutoff = rng.uniform(0.005, 0.03)
        out = _apply_highpass(out, hp_cutoff)

    # 3. Add noise (90% probability)
    if rng.random() < 0.9:
        snr = rng.uniform(8, 30)
        out = _add_noise(out, snr, rng)

    # 4. Power-line hum (50% probability, stronger)
    if rng.random() < 0.5:
        freq = rng.choice([50.0, 60.0])
        amp = rng.uniform(0.01, 0.06)
        out = _add_hum(out, freq, sample_rate, amp)

    # 5. Clipping / saturation (40% probability)
    if rng.random() < 0.4:
        thresh = rng.uniform(0.2, 0.7)
        out = _apply_clipping(out, thresh)

    # 6. Lossy compression artifact (60% probability)
    if rng.random() < 0.6:
        out = _apply_mp3_artifact(out, rng)

    # 7. Reverb (35% probability)
    if rng.random() < 0.35:
        out = _apply_reverb(out, rng, sample_rate)

    return out.clamp(-1.0, 1.0).to(waveform.dtype)
