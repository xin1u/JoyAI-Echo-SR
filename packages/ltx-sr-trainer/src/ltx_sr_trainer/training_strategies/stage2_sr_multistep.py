"""Multi-step stage-2 SR training strategy with fixed-point flow-matching loss.

Instead of training at a single sigma (one-step), this strategy randomly
samples from a fixed set of sigma values (matching the 3-step Euler
denoising schedule used by the distilled pipeline) so the fine-tuned LoRA
maintains multi-step denoising capability.

The noising formula remains the same as the online strategy:
    x_noisy = sigma * noise + (1 - sigma) * lq_condition
    velocity_target = (x_noisy - x_clean) / sigma
"""

from __future__ import annotations

import random as py_random
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class Stage2SRMultistepConfig(TrainingStrategyConfigBase):
    """Configuration for multi-step stage-2 SR training."""

    name: Literal["stage2_sr_multistep"] = "stage2_sr_multistep"

    sigma_schedule: list[float] = Field(
        default=[0.909375, 0.725, 0.421875],
        description=(
            "Fixed sigma points for training. Each step randomly samples "
            "one sigma from this list. These should match the Euler "
            "denoising schedule used at inference time."
        ),
    )
    velocity_loss_weight: float = Field(
        default=1.0,
        description="Weight for velocity prediction MSE loss.",
        ge=0.0,
    )
    x0_loss_weight: float = Field(
        default=0.25,
        description="Weight for direct x0 reconstruction loss.",
        ge=0.0,
    )
    condition_noise_min: float = Field(
        default=0.0,
        description="Minimum noise fraction mixed into LQ condition latent.",
    )
    condition_noise_max: float = Field(
        default=0.0,
        description="Maximum noise fraction mixed into LQ condition latent.",
    )
    condition_drop_prob_min: float = Field(
        default=0.0,
        description="Minimum probability of dropping the LQ condition.",
    )
    condition_drop_prob_max: float = Field(
        default=0.1,
        description="Maximum probability of dropping the LQ condition.",
    )


@dataclass(kw_only=True)
class Stage2SRMultistepModelInputs(ModelInputs):
    video_clean_targets: Tensor


class Stage2SRMultistepStrategy(TrainingStrategy):
    """Multi-step SR strategy with fixed-point sigma sampling."""

    config: Stage2SRMultistepConfig

    def __init__(self, config: Stage2SRMultistepConfig) -> None:
        super().__init__(config)

    def get_data_sources(self) -> dict[str, str]:
        raise NotImplementedError(
            "Stage2SRMultistepStrategy is used with the standalone VideoSRDataset, "
            "not upstream PrecomputedDataset."
        )

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        _timestep_sampler: TimestepSampler,
    ) -> Stage2SRMultistepModelInputs:
        """Build model inputs with per-sample sigma randomly drawn from sigma_schedule.

        Expected batch keys:
            - ``gt_latent``:  (B, C, F', H', W')
            - ``lq_latent``:  (B, C, F', H', W')
            - ``conditions``: dict with video_prompt_embeds, prompt_attention_mask
            - ``fps``:        (B,) or None
        """
        cfg: Stage2SRMultistepConfig = self.config

        gt_latent = batch["gt_latent"]
        lq_latent = batch["lq_latent"]

        device = gt_latent.device
        dtype = gt_latent.dtype
        batch_size = gt_latent.shape[0]

        noise = torch.randn_like(gt_latent)

        # --- condition augmentation (per-sample) ---
        lq_aug = lq_latent.clone()
        for i in range(batch_size):
            noise_scale = py_random.uniform(
                cfg.condition_noise_min, cfg.condition_noise_max
            )
            lq_aug[i] = (1 - noise_scale) * lq_aug[i] + noise_scale * noise[i]

            drop_prob = py_random.uniform(
                cfg.condition_drop_prob_min, cfg.condition_drop_prob_max
            )
            if py_random.random() < drop_prob:
                lq_aug[i].zero_()

        # --- per-sample sigma from fixed schedule ---
        sigma_values = cfg.sigma_schedule
        sigmas = torch.tensor(
            [py_random.choice(sigma_values) for _ in range(batch_size)],
            device=device,
            dtype=dtype,
        )

        # --- noising: x_noisy = sigma * noise + (1 - sigma) * lq_aug ---
        sigmas_5d = sigmas.view(-1, 1, 1, 1, 1)
        x_noisy = sigmas_5d * noise + (1 - sigmas_5d) * lq_aug

        # --- patchify ---
        video_in = self._video_patchifier.patchify(x_noisy)
        gt_patchified = self._video_patchifier.patchify(gt_latent)

        num_frames = gt_latent.shape[2]
        height = gt_latent.shape[3]
        width = gt_latent.shape[4]

        fps = batch.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                "Different FPS values in batch. Found: %s, using first: %s",
                fps.tolist(),
                fps[0].item(),
            )
        fps_val = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        video_seq_len = video_in.shape[1]

        # velocity_target = (x_noisy - x_clean) / sigma  (per-sample)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        velocity_targets = (video_in - gt_patchified) / sigmas_expanded

        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps_val,
            device=device,
            dtype=dtype,
        )
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=video_in,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        return Stage2SRMultistepModelInputs(
            video=video_modality,
            audio=None,
            video_targets=velocity_targets,
            audio_targets=None,
            video_loss_mask=torch.ones(
                batch_size, video_seq_len, dtype=torch.bool, device=device
            ),
            audio_loss_mask=None,
            ref_seq_len=None,
            video_clean_targets=gt_patchified,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: Stage2SRMultistepModelInputs,
    ) -> Tensor:
        loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()

        velocity_loss = (video_pred - inputs.video_targets).pow(2)
        velocity_loss = velocity_loss.mul(loss_mask).div(loss_mask.mean())
        velocity_loss = velocity_loss.mean()

        if self.config.x0_loss_weight <= 0:
            return self.config.velocity_loss_weight * velocity_loss

        sigma = inputs.video.sigma.to(video_pred.dtype).view(-1, 1, 1)
        pred_clean = inputs.video.latent - video_pred * sigma
        clean_loss = (pred_clean - inputs.video_clean_targets).pow(2)
        clean_loss = clean_loss.mul(loss_mask).div(loss_mask.mean())
        clean_loss = clean_loss.mean()

        return (
            self.config.velocity_loss_weight * velocity_loss
            + self.config.x0_loss_weight * clean_loss
        )
