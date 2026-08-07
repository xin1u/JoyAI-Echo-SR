"""Resolution-independent 2x latent upsampling condition projector.

Maps LQ latent [B, C, T, H, W] → HQ token space [B, T*2H*2W, inner_dim]
via learnable convolution-based 2x spatial upsampling.

Unlike CondSRPatchifyProj (fixed lq_h×lq_w → hq_h×hq_w linear),
this module uses conv-based upsampling that generalizes to ANY input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CondLatent2xProj(nn.Module):
    """Resolution-independent 2x latent upsampling + projection to transformer tokens.

    Architecture:
      1. Conv2d feature refinement at LQ resolution (residual block)
      2. PixelShuffle 2x spatial upsampling (learned, resolution-independent)
      3. Conv2d post-refinement at HQ resolution
      4. Channel projection: Linear(in_channels, inner_dim)

    Works for ANY spatial resolution — no hardcoded H×W dimensions.

    Args:
        in_channels: Latent channel dim (128 for LTX VAE).
        inner_dim: Transformer hidden dim (4096 for LTX-2.3 22B video).
        mid_channels: Internal feature channels (default 128).
    """

    def __init__(
        self,
        in_channels: int = 128,
        inner_dim: int = 4096,
        mid_channels: int = 128,
    ) -> None:
        super().__init__()

        # Pre-upsample: refine LQ features
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
        )

        # Learned 2x spatial upsampling via PixelShuffle
        # Conv outputs 4x channels, then pixel_shuffle(2) → 2x spatial
        self.upsample = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )

        # Post-upsample: refine HQ features
        self.post_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

        # Project to transformer hidden dim
        self.proj = nn.Linear(in_channels, inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] raw LQ latent (5D).
        Returns:
            [B, T*2H*2W, inner_dim] tokens in 2x upsampled spatial grid.
        """
        B, C, T, H, W = x.shape

        # Per-frame processing
        x = rearrange(x, "b c t h w -> (b t) c h w")

        # Refine at LQ resolution (residual)
        x = x + self.pre_conv(x)

        # 2x spatial upsample (resolution-independent)
        x = self.upsample(x)  # [(B*T), mid_ch, 2H, 2W]

        # Refine at HQ resolution (residual)
        x = x + self.post_conv(x)  # [(B*T), C, 2H, 2W]

        # Flatten to token sequence
        x = rearrange(x, "(b t) c h w -> b (t h w) c", b=B, t=T)

        # Project to transformer hidden dim
        x = self.proj(x)  # [B, T*2H*2W, inner_dim]
        return x

    @staticmethod
    def init_near_zero(module: "CondLatent2xProj", proj_std: float = 1e-6) -> None:
        """Initialize for stable training start.

        - pre_conv/upsample/post_conv: default init (residual = ~identity at start)
        - proj: near-zero so condition starts with minimal effect
        """
        nn.init.normal_(module.proj.weight, std=proj_std)
        nn.init.zeros_(module.proj.bias)
