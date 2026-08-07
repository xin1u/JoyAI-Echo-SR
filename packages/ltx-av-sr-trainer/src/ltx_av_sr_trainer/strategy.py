"""Audio-Video Restoration training strategy with additive condition injection.

Pipeline:
  HQ video -> VAE encode -> HQ latent (target)
  HQ video -> degrade -> LQ video -> VAE encode -> LQ latent (condition)
  LQ latent -> patchify -> cond_video_proj -> ADD to patchified noisy HQ -> transformer -> loss

  Same for audio: HQ audio -> encode -> HQ latent, degrade -> encode -> LQ latent -> cond_audio_proj -> ADD

CFG follows LTX-2.3's standard text-based CFG (applied at inference only, not during training).
LQ condition uses random dropout (~40%) during training to preserve the model's
text-to-AV generation ability — when LQ is dropped, the model falls back to
pure text-conditioned generation, preventing over-reliance on the LQ signal.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_av_sr_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class AVRestorationConfig(TrainingStrategyConfigBase):
    """Configuration for AV restoration training strategy."""

    name: Literal["av_restoration"] = "av_restoration"

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio restoration",
    )

    lq_drop_prob: float = Field(
        default=0.4,
        description="Probability of dropping LQ condition during training to preserve T2AV ability",
        ge=0.0,
        le=1.0,
    )

    condition_noise_min: float = Field(
        default=0.0,
        description="Minimum noise level added to video condition latents",
        ge=0.0,
    )

    condition_noise_max: float = Field(
        default=0.2,
        description="Maximum noise level added to video condition latents",
        ge=0.0,
    )

    audio_condition_noise_min: float | None = Field(
        default=None,
        description="Minimum noise level added to audio condition latents (None = use condition_noise_min)",
        ge=0.0,
    )

    audio_condition_noise_max: float | None = Field(
        default=None,
        description="Maximum noise level added to audio condition latents (None = use condition_noise_max)",
        ge=0.0,
    )

    audio_loss_weight: float = Field(
        default=1.0,
        description="Weight for audio loss to compensate gradient imbalance from video token dominance",
        gt=0.0,
    )

    first_frame_conditioning_p: float = Field(
        default=0.0,
        description="Probability of first-frame conditioning (typically 0 for SR)",
        ge=0.0,
        le=1.0,
    )

    num_val_samples: int = Field(
        default=1,
        description="Number of samples to randomly draw from the dataset for validation",
        ge=1,
    )

    cross_attn_grad_isolation_layer: int | None = Field(
        default=None,
        description="Layer index where cross-attention gradient isolation ends. "
                    "Layers [0, N) have A↔V cross-attn gradients blocked; [N, 48) flow freely. "
                    "None = disabled (all layers have free gradient flow).",
        ge=0,
        le=48,
    )


class AVRestorationStrategy(TrainingStrategy):
    """AV restoration training strategy with additive LQ condition injection.

    The LQ condition is projected to hidden space via cond_video_proj / cond_audio_proj
    and element-wise added after patchify_proj in the transformer.

    LQ dropout (~40%) preserves T2AV ability: when LQ is dropped, the step is
    equivalent to standard text-to-AV training. CFG is text-based (standard
    LTX-2.3 approach) and applied at inference time only.
    """

    config: AVRestorationConfig

    def __init__(self, config: AVRestorationConfig):
        super().__init__(config)
        self._rng = random.Random()

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> dict[str, str]:
        sources = {
            "hq_latents": "hq_latents",
            "lq_latents": "lq_latents",
            "conditions": "conditions",
        }
        if self.config.with_audio:
            sources["audio_hq_latents"] = "audio_hq_latents"
            sources["audio_lq_latents"] = "audio_lq_latents"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs with additive LQ condition injection."""
        hq_data = batch["hq_latents"]
        lq_data = batch["lq_latents"]
        conditions = batch["conditions"]

        hq_latents = hq_data["latents"]
        lq_latents = lq_data["latents"]

        num_frames = hq_data["num_frames"][0].item()
        height = hq_data["height"][0].item()
        width = hq_data["width"][0].item()

        # Cross-resolution SR: LQ has different spatial dims than HQ.
        # Pass raw 5D LQ latent for CondSRPatchifyProj to handle.
        # Same-resolution (v1): patchify LQ normally for Linear cond_proj.
        cross_res_sr = (lq_latents.shape[-2:] != hq_latents.shape[-2:])

        hq_latents = self._video_patchifier.patchify(hq_latents)
        if cross_res_sr:
            # Keep LQ as raw 5D [B, C, T, H_lq, W_lq] for CondSRPatchifyProj
            pass
        else:
            lq_latents = self._video_patchifier.patchify(lq_latents)

        fps = hq_data.get("fps", None)
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = hq_latents.shape[0]
        video_seq_len = hq_latents.shape[1]
        device = hq_latents.device
        dtype = hq_latents.dtype

        # LQ dropout: randomly drop condition to preserve T2AV ability
        drop_cond = self._rng.random() < self.config.lq_drop_prob

        if drop_cond:
            video_cond_latent = torch.zeros_like(lq_latents)
        else:
            video_cond_latent = lq_latents
            # Optionally add noise to condition
            if self.config.condition_noise_max > 0:
                noise_level = self._rng.uniform(self.config.condition_noise_min, self.config.condition_noise_max)
                cond_noise = torch.randn_like(video_cond_latent) * noise_level
                video_cond_latent = video_cond_latent + cond_noise

        # First-frame conditioning mask
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(hq_latents)
        video_noise = torch.randn_like(hq_latents)

        # Flow matching: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * hq_latents + sigmas_expanded * video_noise

        # Conditioning tokens use clean latents
        cond_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(cond_mask_expanded, hq_latents, noisy_video)

        # Velocity prediction target
        video_targets = video_noise - hq_latents

        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            cond_latent=video_cond_latent,
        )

        video_loss_mask = ~video_conditioning_mask

        # Audio
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                drop_cond=drop_cond,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        drop_cond: bool,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        audio_hq = batch["audio_hq_latents"]
        audio_lq = batch["audio_lq_latents"]

        hq_audio_latents = audio_hq["latents"]
        lq_audio_latents = audio_lq["latents"]

        hq_audio_latents = self._audio_patchifier.patchify(hq_audio_latents)
        lq_audio_latents = self._audio_patchifier.patchify(lq_audio_latents)

        audio_seq_len = hq_audio_latents.shape[1]

        if drop_cond:
            audio_cond_latent = torch.zeros_like(lq_audio_latents)
        else:
            audio_cond_latent = lq_audio_latents
            a_noise_min = self.config.audio_condition_noise_min if self.config.audio_condition_noise_min is not None else self.config.condition_noise_min
            a_noise_max = self.config.audio_condition_noise_max if self.config.audio_condition_noise_max is not None else self.config.condition_noise_max
            if a_noise_max > 0:
                noise_level = self._rng.uniform(a_noise_min, a_noise_max)
                audio_cond_latent = audio_cond_latent + torch.randn_like(audio_cond_latent) * noise_level

        audio_noise = torch.randn_like(hq_audio_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * hq_audio_latents + sigmas_expanded * audio_noise

        audio_targets = audio_noise - hq_audio_latents

        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
            cond_latent=audio_cond_latent,
        )

        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio. Returns [B,]."""
        if torch.isnan(video_pred).any():
            logger.error(
                f"NaN in video_pred! shape={list(video_pred.shape)} "
                f"nan_count={torch.isnan(video_pred).sum().item()}"
            )
        if torch.isnan(inputs.video_targets).any():
            logger.error(f"NaN in video_targets!")
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        masked = video_loss.mul(video_loss_mask)
        video_loss = masked.mean(dim=[-2, -1]) / video_loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            self._last_video_loss = video_loss.detach().mean().item()
            self._last_audio_loss = 0.0
            return video_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean(dim=[-2, -1])

        self._last_video_loss = video_loss.detach().mean().item()
        self._last_audio_loss = audio_loss.detach().mean().item()

        return video_loss + self.config.audio_loss_weight * audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "strategy": "av_restoration",
            "lq_drop_prob": self.config.lq_drop_prob,
            "with_audio": self.config.with_audio,
        }
