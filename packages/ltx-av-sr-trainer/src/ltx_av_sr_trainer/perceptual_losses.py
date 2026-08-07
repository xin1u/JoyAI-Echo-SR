"""Perceptual losses for end-to-end distillation training.

Includes:
  - LPIPS (perceptual similarity, VGG-based)
  - Haar wavelet loss (high-frequency detail supervision)
  - Multi-resolution STFT loss (audio spectral detail)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Video: Haar Wavelet Loss
# ---------------------------------------------------------------------------

def _haar_dwt2_spatial(x: Tensor) -> dict[str, Tensor]:
    """Haar wavelet 2D decomposition on spatial dims of (B, C, T, H, W) or (B, C, H, W)."""
    if x.shape[-2] % 2 != 0:
        x = x[..., :-1, :]
    if x.shape[-1] % 2 != 0:
        x = x[..., :, :-1]
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


def compute_wavelet_loss(
    pred: Tensor,
    target: Tensor,
    bands: tuple[str, ...] = ("HH", "HL", "LH"),
) -> Tensor:
    """L1 loss on high-frequency wavelet subbands. Focuses on edges/textures."""
    pred_bands = _haar_dwt2_spatial(pred.float())
    target_bands = _haar_dwt2_spatial(target.float())
    losses = [F.l1_loss(pred_bands[b], target_bands[b]) for b in bands]
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Video: LPIPS Loss
# ---------------------------------------------------------------------------

def load_lpips_model(device: torch.device, net: str = "vgg") -> nn.Module:
    """Load LPIPS model. Set ECHO_SR_LPIPS_DIR to a local weights dir to avoid network download."""
    import lpips
    import os
    local_weights_dir = os.environ.get("ECHO_SR_LPIPS_DIR")
    if local_weights_dir:
        model_path = os.path.join(local_weights_dir, f"{net}.pth")
        # Set TORCH_HOME so lpips finds backbone weights locally
        os.environ["TORCH_HOME"] = local_weights_dir
        model = lpips.LPIPS(net=net, model_path=model_path, verbose=False).to(device)
    else:
        model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def compute_lpips_loss(
    pred: Tensor,
    target: Tensor,
    lpips_model: nn.Module,
) -> Tensor:
    """LPIPS on single frames. Input: [B, C, H, W] in [-1, 1]."""
    pred_01 = (pred.float() + 1) * 0.5
    target_01 = (target.float() + 1) * 0.5
    return lpips_model(pred_01, target_01).mean()


# ---------------------------------------------------------------------------
# Audio: Multi-resolution STFT Loss
# ---------------------------------------------------------------------------

def compute_stft_loss(
    pred: Tensor,
    target: Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: tuple[int, ...] = (128, 256, 512),
    win_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> Tensor:
    """Multi-resolution STFT loss for audio waveforms.

    Input: [B, T] or [B, C, T] waveforms in [-1, 1].
    Computes spectral convergence + log magnitude L1 at multiple resolutions.
    """
    if pred.dim() == 3:
        pred = pred.mean(dim=1)  # stereo → mono
        target = target.mean(dim=1)

    losses = []
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Skip if waveform is too short for this FFT size
        if pred.shape[-1] < win_size:
            continue
        window = torch.hann_window(win_size, device=pred.device)
        pred_stft = torch.stft(pred.float(), fft_size, hop_size, win_size, window, return_complex=True)
        target_stft = torch.stft(target.float(), fft_size, hop_size, win_size, window, return_complex=True)

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        # Spectral convergence
        sc = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-8)
        # Log magnitude L1
        log_mag = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))

        losses.append(sc + log_mag)

    if not losses:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return sum(losses) / len(losses)
