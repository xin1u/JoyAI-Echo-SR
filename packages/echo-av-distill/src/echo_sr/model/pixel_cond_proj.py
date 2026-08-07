"""Pixel-space LQ condition projector for LTX-2.3 AV super-resolution.

Adapted from FlashVSR's Buffer_LQ4x_Proj architecture:
  PixelShuffle3D (space-to-depth) → CausalConv3d stack (temporal compression) → Linear proj

Key adaptations for LTX-2.3:
  - Temporal compression 8x (3 layers of stride_t=2) to match VAE's temporal behavior
  - First-frame repeat padding (7 frames) for causal alignment
  - Spatial: PixelShuffle(1, 16, 16) → h/16, w/16 matches HQ latent spatial grid
  - Output: single [B, T*H*W, 4096] additive injection (not per-layer)

VAE temporal relationship:
  Encode: pixel_T → latent_T = (pixel_T - 1) / 8 + 1
  Decode: latent_T → pixel_T = (latent_T - 1) * 8 + 1
  121 pixel frames → 16 latent temporal frames → 121 pixel frames

Dimension flow (121 frames, 384×640 LQ):
  [B, 3, 121, 384, 640]
  → pad 7 first frames: [B, 3, 128, 384, 640]
  → PixelShuffle3d(1,16,16): [B, 768, 128, 24, 40]
  → CausalConv3d(stride_t=2) ×3: temporal 128→64→32→16
  → [B, hidden, 16, 24, 40]
  → flatten + Linear → [B, 15360, 4096]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm3d(nn.Module):
    """RMS normalization for 5D tensors [B, C, T, H, W]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution: temporal padding only on the past side."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Redistribute padding: temporal gets full causal (left only), spatial symmetric
        self._causal_padding = (
            self.padding[2], self.padding[2],  # width: symmetric
            self.padding[1], self.padding[1],  # height: symmetric
            2 * self.padding[0], 0,            # temporal: all on left (causal)
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self._causal_padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    """Space-to-depth for 3D: collapse (ff, hh, ww) blocks into channel dim."""

    def __init__(self, ff: int, hh: int, ww: int) -> None:
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff, hh=self.hh, ww=self.ww,
        )


class PixelCondProj(nn.Module):
    """Pixel-space LQ condition projector with causal temporal compression.

    Architecture (adapted from FlashVSR for LTX-2.3):
      1. First-frame repeat padding (7 frames) → 121+7=128 frames
      2. PixelShuffle3d(1, 16, 16): spatial space-to-depth → [B, 768, 128, H/16, W/16]
      3. CausalConv3d × 3 (stride_t=2 each): 8x temporal compression
         - Layer 1: kernel (4,3,3) — spatial+temporal feature extraction
         - Layer 2-3: kernel (4,1,1) — temporal compression only
      4. Linear projection to transformer hidden dim

    This mirrors VAE's causal temporal behavior:
      - VAE: 121 pixel frames → 16 latent temporal frames (formula: (T-1)/8 + 1)
      - PixelCondProj: 128 frames (after padding) → 16 temporal outputs (128/8 = 16)

    Design choices:
      - Layer 1 uses (4,3,3) kernel for local spatial interaction within each 16×16 patch
        neighborhood. After PixelShuffle, adjacent spatial positions are neighboring patches;
        the 3×3 conv captures cross-patch boundary features.
      - Layers 2-3 use (4,1,1) kernel (temporal only) to keep params manageable.
        Spatial interaction is deferred to transformer self-attention.
      - Total ~23M params (vs FlashVSR's 287M, because we only need single additive injection,
        not per-layer injection into 30 transformer blocks).

    Args:
        in_channels: Input pixel channels (3 for RGB).
        inner_dim: Transformer video hidden dim (4096 for LTX-2.3 22B).
        hidden_dim1: First conv output channels (spatial+temporal features).
        hidden_dim2: Second conv output channels (temporal compression).
        hidden_dim3: Third conv output channels (temporal compression, feeds proj).
    """

    def __init__(
        self,
        in_channels: int = 3,
        inner_dim: int = 4096,
        hidden_dim1: int = 512,
        hidden_dim2: int = 768,
        hidden_dim3: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_h = 16
        self.patch_w = 16
        pixel_shuffle_dim = in_channels * self.patch_h * self.patch_w  # 3 × 16 × 16 = 768

        self.pixel_shuffle = PixelShuffle3d(1, self.patch_h, self.patch_w)

        # Layer 1: spatial + temporal feature extraction
        # kernel (4,3,3) captures cross-patch spatial structure + temporal context
        self.conv1 = CausalConv3d(pixel_shuffle_dim, hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm1 = RMSNorm3d(hidden_dim1)
        self.act1 = nn.SiLU()

        # Layer 2-3: temporal compression only
        # kernel (4,1,1) — spatial interaction deferred to transformer self-attention
        self.conv2 = CausalConv3d(hidden_dim1, hidden_dim2, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.norm2 = RMSNorm3d(hidden_dim2)
        self.act2 = nn.SiLU()

        self.conv3 = CausalConv3d(hidden_dim2, hidden_dim3, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.norm3 = RMSNorm3d(hidden_dim3)
        self.act3 = nn.SiLU()

        # Project to transformer hidden dim
        self.proj = nn.Linear(hidden_dim3, inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, T, H_lq, W_lq] LQ pixel tensor in [-1, 1].
               T should be 8n+1 (e.g., 121). H divisible by 16, W divisible by 16.

        Returns:
            [B, num_tokens, inner_dim] where num_tokens = ((T-1)/8+1) * (H/16) * (W/16)
            e.g., [B, 16*24*40, 4096] = [B, 15360, 4096] for T=121, H=384, W=640
        """
        B, C, T, H, W = x.shape

        # First-frame repeat padding: mimic VAE's causal behavior
        # VAE: 121 frames → 16 latent frames (formula: (T-1)/8 + 1)
        # We pad 7 frames of frame_0 so that: (T+7) = 128, divisible by 8
        # 3 layers stride_t=2: 128/8 = 16 temporal outputs ← matches VAE's 16 latent frames
        pad_frames = 7
        first_frame = x[:, :, :1, :, :].expand(-1, -1, pad_frames, -1, -1)
        x = torch.cat([first_frame, x], dim=2)  # [B, 3, T+7, H, W]

        # Space-to-depth: [B, 3, 128, H, W] → [B, 768, 128, H/16, W/16]
        x = self.pixel_shuffle(x)

        # Temporal compression: 128 → 64 → 32 → 16
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        # [B, hidden_dim3, 16, H/16, W/16]

        # Flatten to token sequence + project
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = self.proj(x)  # [B, 16*24*40, 4096] = [B, 15360, 4096]

        return x

    @staticmethod
    def init_near_zero(module: "PixelCondProj", std: float = 1e-6) -> None:
        """Near-zero init on final projection for stable training start."""
        nn.init.normal_(module.proj.weight, std=std)
        nn.init.zeros_(module.proj.bias)
