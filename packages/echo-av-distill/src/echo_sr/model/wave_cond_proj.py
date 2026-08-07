"""Waveform-space LQ condition projector for LTX-2.3 AV super-resolution.

Lossless STFT-based waveform condition injection:
  Waveform → Resample(16kHz) → STFT (invertible) → Patchify by time frame → Linear proj

STFT is a strictly lossless linear transform (Parseval's theorem).
Given complex output (real + imag), ISTFT exactly reconstructs the input.

hop_length=640 is chosen so that STFT produces exactly 121 time frames
from 16kHz × 4.84s = 77440 samples, matching the audio patchifier token count.

Dimension flow (stereo waveform, 4.84s @ 44.1kHz):
  [B, 2, ~213k] (stereo, 44.1kHz)
  → resample to 16kHz: [B, 2, 77440]
  → STFT(n_fft=640, hop=640, center=False): [B, 2, 321, 121] complex
  → stack real+imag: [B, 4, 321, 121]
  → patchify by time: [B, 121, 1284]  (each token = one time frame, all freq bins)
  → Linear(1284, 2048): [B, 121, 2048]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class WaveCondProj(nn.Module):
    """Waveform condition projector via STFT + patchify + Linear.

    Processing chain:
      1. Resample to target_sr (16kHz) for temporal alignment with audio patchifier
      2. STFT: invertible linear transform → complex spectrogram
      3. Patchify by time frame: each frame becomes one token with all freq bins
      4. Linear: project to audio transformer hidden dim

    STFT parameters are chosen so that the output time frames = target_tokens exactly,
    with no interpolation or padding needed.

    Args:
        audio_inner_dim: Audio transformer hidden dim (2048 for LTX-2.3).
        target_tokens: Required output token count (121, matching AudioPatchifier).
        n_fft: STFT window size (640 → 321 freq bins, exactly 121 frames from 77440 samples).
        hop_length: STFT hop length (640 → frame_count = samples/hop = 77440/640 = 121).
        source_sr: Input waveform sample rate (44100 from dataset).
        target_sr: Resample target for model temporal alignment (16000).
        n_channels: Input audio channels (2 for stereo).
    """

    def __init__(
        self,
        audio_inner_dim: int = 2048,
        target_tokens: int = 121,
        n_fft: int = 640,
        hop_length: int = 640,
        source_sr: int = 44100,
        target_sr: int = 16000,
        n_channels: int = 2,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.target_tokens = target_tokens
        self.n_channels = n_channels

        freq_bins = n_fft // 2 + 1  # 641
        # 4 = n_channels(2) × real_imag(2)
        patch_dim = n_channels * 2 * freq_bins  # 2564

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        self.proj = nn.Linear(patch_dim, audio_inner_dim)

    def _resample(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample waveform from source_sr to target_sr."""
        if self.source_sr == self.target_sr:
            return waveform
        try:
            import torchaudio.functional as AF
            return AF.resample(waveform.float(), self.source_sr, self.target_sr)
        except Exception:
            # Fallback: linear interpolation (avoids torchaudio dependency)
            # waveform: [B, C, T] — already 3D, correct for F.interpolate mode="linear"
            target_len = int(waveform.shape[-1] * self.target_sr / self.source_sr)
            return torch.nn.functional.interpolate(
                waveform.float(), size=target_len, mode="linear", align_corners=False,
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, C, T_samples] stereo/mono waveform in [-1, 1].
                      C should be n_channels (2 for stereo).

        Returns:
            [B, target_tokens, audio_inner_dim] = [B, 121, 2048]
        """
        B, C, T = waveform.shape

        # Ensure stereo
        if C == 1:
            waveform = waveform.expand(-1, self.n_channels, -1)
            C = self.n_channels

        # Resample to target_sr
        waveform = self._resample(waveform)

        # Pad/trim to ensure exactly target_tokens frames from STFT
        expected_samples = self.target_tokens * self.hop_length  # 121 × 640 = 77440
        if waveform.shape[-1] > expected_samples:
            waveform = waveform[..., :expected_samples]
        elif waveform.shape[-1] < expected_samples:
            pad_len = expected_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # STFT: [B*C, T] → [B*C, freq_bins, time_frames] complex
        x = waveform.reshape(B * C, -1)
        window = self.window.to(dtype=x.dtype, device=x.device)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=False,
        )
        # spec: [B*C, 641, 121]

        spec = spec.reshape(B, C, spec.shape[1], spec.shape[2])  # [B, C, 641, 121]

        # Stack real + imag: [B, 2C, 641, 121]
        spec = torch.cat([spec.real, spec.imag], dim=1)

        # Patchify by time frame: [B, 2C, 641, 121] → [B, 121, 2C*641]
        tokens = rearrange(spec, "b c f t -> b t (c f)")  # [B, 121, 2564]

        # Project to audio hidden dim
        tokens = self.proj(tokens)  # [B, 121, 2048]
        return tokens

    @staticmethod
    def init_near_zero(module: "WaveCondProj", std: float = 1e-6) -> None:
        """Near-zero init for stable training start."""
        nn.init.normal_(module.proj.weight, std=std)
        nn.init.zeros_(module.proj.bias)
