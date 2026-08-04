"""Standalone stage-2 SR one-step distillation strategy.

This strategy is designed for *pair-style* training samples stored as single
``.pt`` files, where each sample already contains the exact stage-2 input
trajectory and the final teacher output. Prompt embeddings are expected to be
encoded online by the training script and passed in ``batch["conditions"]``.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
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


class Stage2SROneStepConfig(TrainingStrategyConfigBase):
    """Configuration for stage-2 SR one-step distillation."""

    name: Literal["stage2_sr_one_step"] = "stage2_sr_one_step"

    distill_loss_weight: float = Field(
        default=1.0,
        description="Weight for velocity distillation loss against the 3-step teacher.",
        ge=0.0,
    )
    x0_loss_weight: float = Field(
        default=0.25,
        description="Optional direct x0 reconstruction loss weight against the teacher final latent.",
        ge=0.0,
    )


@dataclass(kw_only=True)
class Stage2SRModelInputs(ModelInputs):
    video_clean_targets: Tensor


class Stage2SROneStepStrategy(TrainingStrategy):
    """One-step distillation strategy for stage-2 SR LoRA training."""

    config: Stage2SROneStepConfig

    def __init__(self, config: Stage2SROneStepConfig):
        super().__init__(config)

    def get_data_sources(self) -> dict[str, str]:
        raise NotImplementedError(
            "Stage2SROneStepStrategy is intended to be used with the standalone "
            "Stage2SRPairDataset, not upstream PrecomputedDataset."
        )

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        _timestep_sampler: TimestepSampler,
    ) -> Stage2SRModelInputs:
        video_in = self._video_patchifier.patchify(batch["video_in"])
        teacher_latents = self._video_patchifier.patchify(batch["video_target"])

        num_frames = batch["video_target"].shape[2]
        height = batch["video_target"].shape[3]
        width = batch["video_target"].shape[4]

        fps = batch.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                "Different FPS values found in the batch. Found: %s, using the first one: %s",
                fps.tolist(),
                fps[0].item(),
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_in.shape[0]
        video_seq_len = video_in.shape[1]
        device = video_in.device
        dtype = video_in.dtype

        sigmas = batch["sigma0"].to(device=device, dtype=dtype).view(-1)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        teacher_velocity_targets = (video_in - teacher_latents) / sigmas_expanded

        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
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
            latent=video_in,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        return Stage2SRModelInputs(
            video=video_modality,
            audio=None,
            video_targets=teacher_velocity_targets,
            audio_targets=None,
            video_loss_mask=torch.ones(batch_size, video_seq_len, dtype=torch.bool, device=device),
            audio_loss_mask=None,
            ref_seq_len=None,
            video_clean_targets=teacher_latents,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: Stage2SRModelInputs,
    ) -> Tensor:
        loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()

        distill_loss = (video_pred - inputs.video_targets).pow(2)
        distill_loss = distill_loss.mul(loss_mask).div(loss_mask.mean())
        distill_loss = distill_loss.mean()

        if self.config.x0_loss_weight <= 0:
            return self.config.distill_loss_weight * distill_loss

        sigma = inputs.video.sigma.to(video_pred.dtype).view(-1, 1, 1)
        pred_clean = inputs.video.latent - video_pred * sigma
        clean_loss = (pred_clean - inputs.video_clean_targets).pow(2)
        clean_loss = clean_loss.mul(loss_mask).div(loss_mask.mean())
        clean_loss = clean_loss.mean()
        return self.config.distill_loss_weight * distill_loss + self.config.x0_loss_weight * clean_loss
