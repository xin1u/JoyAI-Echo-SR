#!/usr/bin/env python3
"""Training script for AV restoration with additive condition injection.

Wraps the LTX trainer with the AVRestorationStrategy and initializes
the cond_video_proj / cond_audio_proj layers on the transformer.

Key flow:
  1. Load base model (ltx-2.3-22b-dev.safetensors)
  2. Add LoRA adapters (structure from distilled-lora-384, weights init to zero)
  3. Init cond_video_proj + cond_audio_proj (near-zero, trainable)
  4. Optionally load SR checkpoint for resume (load_checkpoint)
  5. Train LoRA + cond_proj, everything else frozen
  6. Save merged checkpoint (LoRA + cond_proj) for single-file inference

Usage:
    python scripts/train.py configs/ltx23_av_restoration_lora.yaml

    # Multi-GPU
    accelerate launch scripts/train.py configs/ltx23_av_restoration_lora.yaml
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _pkg in ("ltx-core-1.1", "ltx-trainer-1.1", "ltx-av-sr-trainer"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from ltx_av_sr_trainer import logger


# ---------------------------------------------------------------------------
# Strategy Registry Patch
# ---------------------------------------------------------------------------

def patch_strategy_registry() -> None:
    """Register AVRestorationStrategy into the trainer's strategy factory."""
    import ltx_trainer.training_strategies as strategies
    from ltx_av_sr_trainer.strategy import AVRestorationConfig, AVRestorationStrategy

    original_factory = strategies.get_training_strategy

    def patched_factory(config):
        if isinstance(config, AVRestorationConfig):
            strategy = AVRestorationStrategy(config)
            logger.info(
                "Using AVRestorationStrategy (audio enabled)"
                if config.with_audio
                else "Using AVRestorationStrategy (audio disabled)"
            )
            return strategy
        return original_factory(config)

    strategies.get_training_strategy = patched_factory


# ---------------------------------------------------------------------------
# SRTrainer: subclass of LtxvTrainer for AV restoration
# ---------------------------------------------------------------------------
# Handles:
#   - Mixed-rank LoRA matching the distilled checkpoint exactly
#   - cond_video_proj / cond_audio_proj (full-param, non-LoRA) for LQ injection
#   - Unified save/load of LoRA + cond_proj in one safetensors file

from peft import LoraConfig as PeftLoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor


def _read_lora_spec_from_checkpoint(ckpt_path: str) -> tuple[list[str], int, dict[str, int]]:
    """Read a LoRA checkpoint and return (target_module_names, default_rank, rank_pattern).

    Scans all lora_A tensors to discover module names and per-module ranks.
    Returns module names as full paths within the transformer (suitable for
    PEFT regex matching), the most common rank as default, and a rank_pattern
    dict for modules that differ.
    """
    import re as _re
    from collections import Counter

    sd = load_file(ckpt_path)
    module_ranks: dict[str, int] = {}
    for key, tensor in sd.items():
        if ".lora_A." not in key:
            continue
        # "diffusion_model.xxx.lora_A.weight" → "xxx"
        name = key.replace("diffusion_model.", "", 1)
        name = _re.sub(r"\.lora_A\.weight$", "", name)
        module_ranks[name] = tensor.shape[0]  # rank = lora_A rows

    # Most common rank → default; others go into rank_pattern
    rank_counts = Counter(module_ranks.values())
    default_rank = rank_counts.most_common(1)[0][0]
    rank_pattern = {name: r for name, r in module_ranks.items() if r != default_rank}

    module_names = sorted(module_ranks.keys())
    logger.info(
        f"Checkpoint LoRA spec: {len(module_names)} modules, "
        f"default rank={default_rank}, {len(rank_pattern)} overrides"
    )
    return module_names, default_rank, rank_pattern


class SRTrainer:
    """Overrides LtxvTrainer methods for AV restoration training.

    Handles mixed-rank LoRA (auto-detected from checkpoint) plus
    cond_video_proj / cond_audio_proj (full-param, non-LoRA).
    Everything is saved into a single safetensors checkpoint.
    """

    @staticmethod
    def _get_base(trainer) -> Any:
        t = trainer._transformer
        if hasattr(t, "module"):
            t = t.module
        if hasattr(t, "get_base_model"):
            t = t.get_base_model()
        return t

    @staticmethod
    def apply(TrainerCls, sr_spatial=None, lora_structure_from: str | None = None):

        # ---- 1. _setup_lora: auto-detect modules/ranks from checkpoint ----
        def _setup_lora(self) -> None:
            base = self._transformer

            # Init cond_proj BEFORE PEFT so they exist as regular nn.Linear
            if sr_spatial is not None:
                # Cross-resolution SR: use learned spatial patchify for video
                base.init_cond_sr_proj(
                    lq_h=sr_spatial["lq_h"], lq_w=sr_spatial["lq_w"],
                    hq_h=sr_spatial["hq_h"], hq_w=sr_spatial["hq_w"],
                    init_std=1e-6,
                )
                logger.info(
                    f"Initialized cond_sr_video_proj "
                    f"(LQ {sr_spatial['lq_h']}x{sr_spatial['lq_w']} → "
                    f"HQ {sr_spatial['hq_h']}x{sr_spatial['hq_w']}) and cond_audio_proj (std=1e-6)"
                )
            else:
                base.init_cond_proj(init_std=1e-6)
                logger.info("Initialized cond_video_proj and cond_audio_proj (std=1e-6)")

            # Auto-detect LoRA structure from reference checkpoint (for parameter
            # discovery only — weights are NOT loaded here).  Falls back to
            # load_checkpoint or YAML lora config when not specified.
            structure_ckpt = lora_structure_from or self._config.model.load_checkpoint
            if structure_ckpt and Path(structure_ckpt).exists():
                module_names, default_rank, rank_pattern = _read_lora_spec_from_checkpoint(structure_ckpt)
                alpha_pattern = {k: v for k, v in rank_pattern.items()}
                logger.info(f"LoRA structure from: {structure_ckpt}")
            else:
                cfg = self._config.lora
                module_names = cfg.target_modules
                default_rank = cfg.rank
                rank_pattern = {}
                alpha_pattern = {}

            lora_config = PeftLoraConfig(
                r=default_rank,
                lora_alpha=default_rank,
                target_modules=module_names,
                lora_dropout=self._config.lora.dropout,
                init_lora_weights=True,
                rank_pattern=rank_pattern,
                alpha_pattern=alpha_pattern,
            )
            self._transformer = get_peft_model(self._transformer, lora_config)

            # Rebuild preprocessors so they reference the LoRA-wrapped patchify_proj
            # instead of the stale pre-PEFT nn.Linear (otherwise LoRA on patchify_proj
            # and audio_patchify_proj is silently bypassed during forward pass).
            peft_base = self._transformer.get_base_model()
            peft_base._rebuild_preprocessors_with_cond(use_sr_proj=(sr_spatial is not None))

        # ---- 2. _collect_trainable_params: LoRA + cond_proj ----
        def _collect_trainable_params(self) -> None:
            if self._config.model.training_mode == "lora":
                _setup_lora(self)
            else:
                self._transformer.requires_grad_(True)

            base = self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
            for name, param in base.named_parameters():
                if "cond_" in name and "proj" in name:
                    param.requires_grad = True

            self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
            total = sum(p.numel() for p in self._trainable_params)
            self._total_trainable_params = total
            logger.info(f"Trainable params (LoRA + cond_proj): {total:,}")

        # ---- 3. _load_lora_checkpoint: load LoRA + cond_proj ----
        def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
            state_dict = load_file(checkpoint_path)
            # Checkpoint keys: "diffusion_model.xxx.lora_A.weight"
            # set_peft_model_state_dict expects: "base_model.model.xxx.lora_A.weight"
            remapped = {}
            for k, v in state_dict.items():
                k = k.replace("diffusion_model.", "", 1)
                remapped[f"base_model.model.{k}"] = v

            cond_keys = {k for k in remapped if "cond_" in k and "proj" in k}
            cond_state = {k: remapped[k] for k in cond_keys}
            lora_state = {k: v for k, v in remapped.items() if k not in cond_keys}

            set_peft_model_state_dict(self._transformer, lora_state)

            if cond_state:
                base = self._transformer.get_base_model()
                for name, param in base.named_parameters():
                    full_key = f"base_model.model.{name}"
                    if full_key in cond_state:
                        param.data.copy_(cond_state[full_key].to(param.dtype))
                logger.info(f"Restored {len(cond_state)} cond_proj tensors from checkpoint")

            logger.info("LoRA + cond_proj checkpoint loaded successfully")

        # ---- 4. _save_checkpoint: save LoRA + cond_proj as one safetensors ----
        def _save_checkpoint(self) -> Path | None:
            from accelerate.utils import DistributedType
            is_lora = self._config.model.training_mode == "lora"
            is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
            is_main = self._accelerator.is_main_process

            save_dir = Path(self._config.output_dir) / "checkpoints"
            prefix = "lora" if is_lora else "model"
            filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
            saved_weights_path = save_dir / filename

            self._accelerator.wait_for_everyone()
            full_state_dict = self._accelerator.get_state_dict(self._transformer)

            if not is_main:
                return None

            save_dir.mkdir(exist_ok=True, parents=True)
            save_dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32

            if is_lora:
                unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
                lora_sd = get_peft_model_state_dict(unwrapped, state_dict=full_state_dict if is_fsdp else None)
                lora_sd = {k.replace("base_model.model.", "", 1): v for k, v in lora_sd.items()}
                lora_sd = {f"diffusion_model.{k}": v for k, v in lora_sd.items()}

                cond_state = {}
                for k, v in full_state_dict.items():
                    if "_cond_" in k and "proj" in k:
                        clean_k = k.replace("base_model.model.", "", 1)
                        cond_state[f"diffusion_model.{clean_k}"] = v.detach().cpu()

                state_dict = {**lora_sd, **cond_state}
                state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()}

                metadata = self._build_checkpoint_metadata()
                metadata["has_cond_proj"] = "true"
                save_file(state_dict, saved_weights_path, metadata=metadata)

                logger.info(
                    f"Saved {len(lora_sd)} LoRA + {len(cond_state)} cond_proj tensors "
                    f"to {saved_weights_path.name}"
                )
            else:
                full_state_dict = {k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in full_state_dict.items()}
                self._accelerator.save(full_state_dict, saved_weights_path)

            rel_path = saved_weights_path.relative_to(self._config.output_dir)
            logger.info(f"Checkpoint step {self._global_step} saved: {rel_path}")

            # Save TinyDecoder if trainable
            _td = getattr(self, "_tiny_decoder", None)
            if _td is not None and any(p.requires_grad for p in _td.parameters()):
                td_path = save_dir / f"tiny_decoder_step_{self._global_step:05d}.pth"
                torch.save(_td.state_dict(), td_path)
                logger.info(f"TinyDecoder saved: {td_path.name}")

            self._checkpoint_paths.append(saved_weights_path)
            self._cleanup_checkpoints()
            self._save_training_state(save_dir)

            return saved_weights_path

        # ---- 5. _prepare_models_for_training: ensure cond_proj on GPU ----
        _original_prepare = TrainerCls._prepare_models_for_training

        def _prepare_models_for_training(self) -> None:
            _original_prepare(self)

            # Move cond_proj to GPU — they're root-level, not inside FSDP-wrapped blocks
            device = self._accelerator.device
            wrapped_base = SRTrainer._get_base(self)
            for name, param in wrapped_base.named_parameters():
                if "cond_" in name and "proj" in name:
                    param.data = param.data.to(device=device)
                    param.requires_grad = True

        # ---- 6. _training_step wrapper: NaN/Inf detection with diagnostics ----
        _original_training_step = TrainerCls._training_step

        def _nan_check(t, name):
            """Check a tensor for NaN/Inf and log diagnostics."""
            if t is None:
                return False
            if isinstance(t, torch.Tensor):
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                if has_nan or has_inf:
                    logger.error(
                        f"  NaN/Inf in {name}: shape={list(t.shape)} "
                        f"nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()} "
                        f"min={t[~torch.isnan(t)].min().item() if (~torch.isnan(t)).any() else 'all-nan'} "
                        f"max={t[~torch.isnan(t)].max().item() if (~torch.isnan(t)).any() else 'all-nan'}"
                    )
                    return True
            return False

        def _training_step_with_nan_guard(self, batch):
            # Save a snapshot of conditions BEFORE _original_training_step mutates them
            conds = batch.get("conditions")
            if conds is not None:
                self._last_train_batch = {
                    **batch,
                    "conditions": {k: v.clone() if isinstance(v, Tensor) else v for k, v in conds.items()},
                }
            else:
                self._last_train_batch = batch

            output = _original_training_step(self, batch)
            loss_val = output.loss.detach().mean()
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                logger.warning(
                    f"Step {self._global_step}: NaN/Inf loss detected ({loss_val.item():.4f}), "
                    f"replacing with zero loss to skip this step"
                )
                # Extra diagnostics: check cond_proj params
                base = SRTrainer._get_base(self)
                for name, param in base.named_parameters():
                    if "cond_" in name and "proj" in name:
                        _nan_check(param.data, f"param:{name}")
                        logger.info(
                            f"  param:{name}: shape={list(param.shape)} "
                            f"device={param.device} dtype={param.dtype} "
                            f"absmax={param.data.abs().max().item():.6f}"
                        )

                zero_loss = torch.zeros_like(output.loss)
                zero_loss.requires_grad = True
                from dataclasses import replace as _replace
                output = _replace(output, loss=zero_loss)
                if not hasattr(self, "_nan_count"):
                    self._nan_count = 0
                self._nan_count += 1
                if self._nan_count % 10 == 0:
                    logger.error(f"NaN/Inf loss count reached {self._nan_count}")
            return output

        # ---- Apply ----
        TrainerCls._lora_structure_from = lora_structure_from
        TrainerCls._setup_lora = _setup_lora
        TrainerCls._collect_trainable_params = _collect_trainable_params
        TrainerCls._load_lora_checkpoint = _load_lora_checkpoint
        TrainerCls._save_checkpoint = _save_checkpoint
        TrainerCls._prepare_models_for_training = _prepare_models_for_training
        TrainerCls._training_step = _training_step_with_nan_guard


# ---------------------------------------------------------------------------
# Validation Patch: AV Restoration from Real Data
# ---------------------------------------------------------------------------

def _caption_to_slug(caption: str, max_len: int = 40) -> str:
    """Turn a caption into a short filesystem-safe slug for file naming."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9一-鿿]+", "_", caption).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "unknown"


def patch_validation(trainer, val_data_root: str | None, lq_dims: tuple[int, int, int] | None = None) -> None:
    """Override _sample_videos to do AV restoration on real data samples.

    Number of samples controlled by ``training_strategy.num_val_samples`` (yaml).

    Per sample saves (distinguished by caption slug):
      - step_XXXXXX_{slug}_lq.mp4        LQ input video (with audio)
      - step_XXXXXX_{slug}_restored.mp4   Restored output video (with audio)
      - step_XXXXXX_{slug}_compare.png    Mid-frame side-by-side (LQ | HQ)
    """
    if val_data_root is None or not Path(val_data_root).is_dir():
        logger.info("Validation data root not found, keeping default T2AV validation")
        return

    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider, STGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import (
        AudioPatchifier,
        VideoLatentPatchifier,
    )
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )
    from ltx_core.model.audio_vae import AudioProcessor, encode_audio
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import (
        Audio,
        AudioLatentShape,
        LatentState,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
        VideoPixelShape,
    )
    from ltx_av_sr_trainer.raw_av_dataset import RawAVDataset
    from ltx_trainer.gpu_utils import free_gpu_memory
    from ltx_trainer.model_loader import load_audio_vae_encoder
    from ltx_trainer.video_utils import save_video

    VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

    val_config = trainer._config.validation
    strategy_config = trainer._config.training_strategy
    num_val_samples = getattr(strategy_config, "num_val_samples", 1)
    width, height, num_frames = val_config.video_dims

    # LQ resolution for cross-res SR validation
    if lq_dims is not None:
        lq_width, lq_height, _ = lq_dims
    else:
        lq_width, lq_height = width, height
    is_cross_res_mode = (lq_width != width or lq_height != height)

    val_dataset = RawAVDataset(
        data_root=val_data_root,
        target_width=lq_width if is_cross_res_mode else width,
        target_height=lq_height if is_cross_res_mode else height,
        target_frames=num_frames,
        audio_sample_rate=44100,
        caption_lang="en",
        require_audio=val_config.generate_audio,
    )

    logger.info(
        f"AV Restoration validation: {num_val_samples} sample(s) from "
        f"{val_data_root} ({len(val_dataset)} total)"
    )

    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    # Audio encoder loaded lazily on first validation (kept across runs)
    _audio_enc_cache: dict[str, Any] = {}

    # Alternating validation toggle: 1 → [TRAIN] (current batch), 0 → [VAL] (random)
    _val_state: dict[str, Any] = {"toggle": 1}

    def _ensure_audio_encoder(device: torch.device) -> None:
        """Lazy-load audio encoder on first use."""
        if "encoder" in _audio_enc_cache:
            return
        model_path = trainer._config.model.model_path
        _audio_enc_cache["encoder"] = load_audio_vae_encoder(
            model_path, device=device, dtype=torch.bfloat16
        )
        _audio_enc_cache["encoder"].eval()
        enc = _audio_enc_cache["encoder"]
        _audio_enc_cache["processor"] = AudioProcessor(
            target_sample_rate=enc.sample_rate,
            mel_bins=enc.mel_bins,
            mel_hop_length=enc.mel_hop_length,
            n_fft=enc.n_fft,
        ).to(device=device)
        logger.info(f"Loaded AudioEncoder for validation (sr={enc.sample_rate})")

    def _encode_lq_audio(
        sample: dict, device: torch.device, generate_audio: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Encode LQ audio waveform → patchified tokens. Returns (tokens, waveform, sr)."""
        if not generate_audio or "audio" not in sample:
            return None, None, 44100
        try:
            _ensure_audio_encoder(device)
            waveform = sample["audio"]  # (C, T) or (T,)
            sr = int(sample.get("audio_sample_rate", 44100))
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # (1, T)
            if waveform.shape[0] == 1:
                waveform = waveform.expand(2, -1)  # mono → stereo
            audio_obj = Audio(
                waveform=waveform.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                sampling_rate=sr,
            )
            latent = encode_audio(audio_obj, _audio_enc_cache["encoder"], _audio_enc_cache["processor"])
            tokens = audio_patchifier.patchify(latent)
            return tokens, waveform, sr
        except Exception as e:
            logger.warning(f"Audio encode failed: {e}")
            return None, None, 44100

    def _get_text_embeddings(device: torch.device):
        """Get cached text embeddings (positive + negative)."""
        if trainer._cached_validation_embeddings and len(trainer._cached_validation_embeddings) > 0:
            cached = trainer._cached_validation_embeddings[0]
            v_pos = cached.video_context_positive.to(device)
            a_pos = cached.audio_context_positive.to(device)
            v_neg = (
                cached.video_context_negative.to(device)
                if cached.video_context_negative is not None else None
            )
            a_neg = (
                cached.audio_context_negative.to(device)
                if cached.audio_context_negative is not None else None
            )
        else:
            v_pos = torch.zeros(1, 1, 4096, device=device, dtype=torch.bfloat16)
            a_pos = torch.zeros(1, 1, 2048, device=device, dtype=torch.bfloat16)
            v_neg = torch.zeros(1, 1, 4096, device=device, dtype=torch.bfloat16)
            a_neg = torch.zeros(1, 1, 2048, device=device, dtype=torch.bfloat16)
        return v_pos, a_pos, v_neg, a_neg

    def _build_stg_perturbation_config(val_cfg) -> BatchedPerturbationConfig | None:
        """Build STG perturbation config from validation config."""
        stg_blocks = getattr(val_cfg, "stg_blocks", None)
        stg_mode = getattr(val_cfg, "stg_mode", "stg_av")
        if not stg_blocks:
            return None
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks)
        ]
        if stg_mode == "stg_av":
            perturbations.append(
                Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks)
            )
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])

    def _load_sample_text_embeddings(
        uuid: str, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Load per-sample text embeddings from preprocessed conditions cache.

        Falls back to cached validation embeddings (generic prompt) if per-sample
        conditions are unavailable.
        """
        cache_root = Path(trainer._config.data.preprocessed_data_root)
        cond_path = cache_root / "conditions" / f"{uuid}.pt"

        if cond_path.exists() and trainer._embeddings_processor is not None:
            from ltx_core.text_encoders.gemma import convert_to_additive_mask

            cond_data = torch.load(cond_path, weights_only=False)
            video_features = cond_data["video_prompt_embeds"].unsqueeze(0).to(device=device)
            audio_features = cond_data.get("audio_prompt_embeds")
            if audio_features is not None:
                audio_features = audio_features.unsqueeze(0).to(device=device)
            mask = cond_data["prompt_attention_mask"].unsqueeze(0).to(device=device)
            additive_mask = convert_to_additive_mask(mask, video_features.dtype)

            v_pos, a_pos, _ = trainer._embeddings_processor.create_embeddings(
                video_features, audio_features, additive_mask,
            )
            _, _, v_neg, a_neg = _get_text_embeddings(device)
            return v_pos, a_pos, v_neg, a_neg

        return _get_text_embeddings(device)

    def _run_restoration(
        lq_video_cond: torch.Tensor,
        lq_audio_tokens: torch.Tensor | None,
        pixel_shape: VideoPixelShape,
        data_idx: int,
        device: torch.device,
        x0_model,
        cfg_guider,
        stg_guider,
        stg_ptb_config: BatchedPerturbationConfig | None,
        val_cfg,
        sampling_ctx,
        text_embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run denoising loop aligned with ValidationSampler._run_denoising()."""
        if text_embeddings is not None:
            v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = text_embeddings
        else:
            v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = _get_text_embeddings(device)

        # Create tools (aligned with ValidationSampler._create_video/audio_latent_tools)
        video_tools = VideoLatentTools(
            patchifier=video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
            fps=pixel_shape.fps,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )
        audio_tools: AudioLatentTools | None = None
        if lq_audio_tokens is not None:
            audio_tools = AudioLatentTools(
                patchifier=audio_patchifier,
                target_shape=AudioLatentShape.from_duration(
                    batch=1, duration=float(pixel_shape.frames) / float(pixel_shape.fps)
                ),
            )

        # Create initial state + noise (aligned with GaussianNoiser)
        generator = torch.Generator(device=device).manual_seed(val_cfg.seed + data_idx)
        noiser = GaussianNoiser(generator)

        video_state = video_tools.create_initial_state(device, torch.bfloat16)
        video_state = noiser(video_state, noise_scale=1.0)

        audio_state: LatentState | None = None
        if audio_tools is not None:
            audio_state = audio_tools.create_initial_state(device, torch.bfloat16)
            audio_state = noiser(audio_state, noise_scale=1.0)

        # Scheduler + stepper — cap sigma_shift at 13.0 to match training sampler
        # (LTX2Scheduler's linear extrapolation overflows for large token counts,
        #  producing NaN sigmas when tokens >> MAX_SHIFT_ANCHOR=4096)
        scheduler = LTX2Scheduler()
        target_shape = video_tools.target_shape
        tokens = target_shape.frames * target_shape.height * target_shape.width
        SHIFT_CAP = 13.0
        _x1, _x2, _base = 1024, 4096, 0.95
        _max_default = 2.05
        raw_shift = (_max_default - _base) * (tokens - _x1) / (_x2 - _x1) + _base

        if val_cfg.inference_steps == 1:
            # 1-step inference: skip scheduler shift/stretch (causes NaN for large token counts)
            # Direct sigma schedule: start at 1.0, end at 0.0
            sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
            logger.info(f"  1-step inference: using raw sigmas=[1.0, 0.0], tokens={tokens}")
        else:
            if raw_shift > SHIFT_CAP:
                safe_max = _base + (SHIFT_CAP - _base) * (_x2 - _x1) / (tokens - _x1)
                logger.info(
                    f"  Scheduler shift capped: raw={raw_shift:.2f} > {SHIFT_CAP}, "
                    f"tokens={tokens}, max_shift={safe_max:.4f} (default={_max_default})"
                )
            else:
                safe_max = _max_default
            dummy_latent = torch.empty(1, 1, target_shape.frames, target_shape.height, target_shape.width, device=device)
            sigmas = scheduler.execute(
                steps=val_cfg.inference_steps, latent=dummy_latent, max_shift=safe_max,
            ).to(device).float()
            if torch.isnan(sigmas).any():
                logger.error(f"  NaN in sigma schedule! sigmas={sigmas.tolist()}")
        stepper = EulerDiffusionStep()

        # Initial modalities
        # Align LQ condition tokens to match target latent sequence lengths
        def _align_tokens(tokens, target_len):
            if tokens is None or tokens.shape[1] == target_len:
                return tokens
            if tokens.shape[1] > target_len:
                return tokens[:, :target_len, :]
            pad = torch.zeros(
                tokens.shape[0], target_len - tokens.shape[1], tokens.shape[2],
                device=tokens.device, dtype=tokens.dtype,
            )
            return torch.cat([tokens, pad], dim=1)

        aligned_lq_video = _align_tokens(lq_video_cond, video_state.latent.shape[1]) if lq_video_cond.dim() == 3 else lq_video_cond

        video_mod = Modality(
            enabled=True,
            latent=video_state.latent,
            sigma=sigmas[0].repeat(video_state.latent.shape[0]),
            timesteps=video_state.denoise_mask,
            positions=video_state.positions,
            context=v_ctx_pos,
            context_mask=None,
            cond_latent=aligned_lq_video,
        )
        audio_mod: Modality | None = None
        if audio_state is not None:
            aligned_lq_audio = _align_tokens(lq_audio_tokens, audio_state.latent.shape[1])
            audio_mod = Modality(
                enabled=True,
                latent=audio_state.latent,
                sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
                timesteps=audio_state.denoise_mask,
                positions=audio_state.positions,
                context=a_ctx_pos,
                context_mask=None,
                cond_latent=aligned_lq_audio,
            )

        # Denoising loop (aligned with validation_sampler.py L528-598)
        for step_idx, sigma in enumerate(sigmas[:-1]):
            video_mod = replace(
                video_mod,
                latent=video_state.latent,
                sigma=sigma.repeat(video_state.latent.shape[0]),
                timesteps=sigma * video_state.denoise_mask,
                positions=video_state.positions,
            )
            if audio_mod is not None and audio_state is not None:
                audio_mod = replace(
                    audio_mod,
                    latent=audio_state.latent,
                    sigma=sigma.repeat(audio_state.latent.shape[0]),
                    timesteps=sigma * audio_state.denoise_mask,
                    positions=audio_state.positions,
                )

            # Positive pass
            denoised_video, denoised_audio = x0_model(video=video_mod, audio=audio_mod, perturbations=None)
            pos_video, pos_audio = denoised_video, denoised_audio

            if torch.isnan(denoised_video).any():
                logger.warning(
                    f"  NaN in denoised_video at step {step_idx}, sigma={sigma:.4f}, "
                    f"input_nan={torch.isnan(video_mod.latent).any().item()}, "
                    f"cond_nan={torch.isnan(video_mod.cond_latent).any().item() if video_mod.cond_latent is not None else False}"
                )

            # CFG
            if cfg_guider.enabled() and v_ctx_neg is not None:
                video_neg = replace(video_mod, context=v_ctx_neg)
                audio_neg = replace(audio_mod, context=a_ctx_neg) if audio_mod is not None else None
                neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)
                denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                if audio_mod is not None and denoised_audio is not None and neg_audio is not None:
                    denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

            # STG
            if stg_guider.enabled() and stg_ptb_config is not None:
                ptb_video, ptb_audio = x0_model(
                    video=video_mod, audio=audio_mod, perturbations=stg_ptb_config
                )
                denoised_video = denoised_video + stg_guider.delta(pos_video, ptb_video)
                if audio_mod is not None and denoised_audio is not None and ptb_audio is not None:
                    denoised_audio = denoised_audio + stg_guider.delta(pos_audio, ptb_audio)

            # Euler step via official stepper
            video_state = replace(
                video_state,
                latent=stepper.step(
                    sample=video_mod.latent, denoised_sample=denoised_video,
                    sigmas=sigmas, step_index=step_idx,
                ),
            )
            if audio_mod is not None and audio_state is not None:
                audio_state = replace(
                    audio_state,
                    latent=stepper.step(
                        sample=audio_mod.latent, denoised_sample=denoised_audio,
                        sigmas=sigmas, step_index=step_idx,
                    ),
                )

            sampling_ctx.advance_step()

        # NaN checkpoint: final denoised latent (before unpatchify)
        _final_latent = video_state.latent
        logger.info(
            f"  Denoising done: latent shape={list(_final_latent.shape)}, "
            f"nan={torch.isnan(_final_latent).any().item()}, "
            f"range=[{_final_latent.min().item():.4f}, {_final_latent.max().item():.4f}]"
        )

        # Decode video via tools.unpatchify() + tiled decode
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        logger.info(
            f"  After unpatchify: shape={list(video_state.latent.shape)}, "
            f"nan={torch.isnan(video_state.latent).any().item()}, "
            f"range=[{video_state.latent.min().item():.4f}, {video_state.latent.max().item():.4f}]"
        )

        from ltx_core.model.video_vae import TilingConfig as _TilingConfig
        dec_tiling = _TilingConfig.default()

        # Use TinyDecoder if available (fast), otherwise full VAE decoder
        _td = getattr(trainer, "_tiny_decoder", None)
        if _td is not None:
            # TinyDecoder: input [N, T, C, H, W], output [N, T_out, 3, H_px, W_px] in [0,1]
            lat_ntchw = video_state.latent.unsqueeze(0) if video_state.latent.dim() == 4 else video_state.latent
            lat_ntchw = lat_ntchw.permute(0, 2, 1, 3, 4).to(torch.bfloat16)  # [B, T, C, H, W]
            vid_01 = _td.decode_video(lat_ntchw, parallel=True, show_progress_bar=False)
            # [B, T, 3, H, W] → [B, 3, T, H, W] and scale to [-1, 1]
            video_out = vid_01.permute(0, 2, 1, 3, 4) * 2 - 1
        else:
            trainer._vae_decoder.to(device)
            video_chunks = list(trainer._vae_decoder.tiled_decode(video_state.latent, dec_tiling))
            for ci, vc in enumerate(video_chunks):
                if torch.isnan(vc).any():
                    logger.warning(f"  NaN in VAE decode chunk {ci}: shape={list(vc.shape)}")
            video_out = torch.cat(video_chunks, dim=2)  # cat along temporal dim
            trainer._vae_decoder.to("cpu")

        # Decode audio
        audio_out = None
        if audio_state is not None and audio_tools is not None and trainer._audio_vae is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            trainer._audio_vae.to(device)
            audio_mel = trainer._audio_vae(audio_state.latent)
            trainer._audio_vae.to("cpu")
            if trainer._vocoder is not None:
                trainer._vocoder.to(device)
                audio_out = trainer._vocoder(audio_mel).squeeze(0).float()
                trainer._vocoder.to("cpu")

        return video_out, audio_out

    @torch.no_grad()
    def _train_mode_validation(
        progress, val_cfg, generate_audio, device, is_main, rank, world_size,
    ) -> list[tuple[int, Path]]:
        """[TRAIN] mode: run restoration inference on current training batch.

        Decodes LQ/HQ from batch latents, runs denoising, saves 3-panel compare.
        """
        from ltx_core.model.video_vae import TilingConfig as _TC
        from ltx_core.text_encoders.gemma import convert_to_additive_mask

        mode_tag = "TRAIN"
        batch = getattr(trainer, "_last_train_batch", None)
        if batch is None:
            logger.warning("[TRAIN] No training batch available — skipping")
            return []

        hq_data = batch["hq_latents"]
        lq_data = batch["lq_latents"]
        conditions = batch["conditions"]

        hq_latent = hq_data["latents"][0:1].to(device=device, dtype=torch.bfloat16)
        lq_latent = lq_data["latents"][0:1].to(device=device, dtype=torch.bfloat16)
        t_num_frames = hq_data["num_frames"][0].item()
        t_height = hq_data["height"][0].item()
        t_width = hq_data["width"][0].item()

        logger.info(
            f"Validation [{mode_tag}] step {trainer._global_step}: "
            f"latent shape={list(hq_latent.shape)}, spatial={t_height}x{t_width}, frames={t_num_frames}"
        )

        # LQ condition: 5D raw latent for cross-res SR, 3D patchified for same-res
        if lq_latent.shape[-2:] != hq_latent.shape[-2:]:
            lq_video_cond = lq_latent
        else:
            lq_video_cond = video_patchifier.patchify(lq_latent)

        # Audio condition from training batch
        lq_audio_tokens = None
        if generate_audio and "audio_lq_latents" in batch:
            alq = batch["audio_lq_latents"]["latents"][0:1].to(device=device, dtype=torch.bfloat16)
            lq_audio_tokens = audio_patchifier.patchify(alq)

        # Text embeddings from batch conditions (apply embeddings processor)
        video_features = conditions["video_prompt_embeds"][0:1].to(device=device)
        audio_features = conditions.get("audio_prompt_embeds")
        if audio_features is not None:
            audio_features = audio_features[0:1].to(device=device)
        mask = conditions["prompt_attention_mask"][0:1].to(device=device)
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)
        v_pos, a_pos, _ = trainer._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask,
        )
        _, _, v_neg, a_neg = _get_text_embeddings(device)
        text_embeddings = (v_pos, a_pos, v_neg, a_neg)

        # Pixel-space dimensions (latent → pixel)
        pixel_h = t_height * VIDEO_SCALE_FACTORS.height
        pixel_w = t_width * VIDEO_SCALE_FACTORS.width
        pixel_f = (t_num_frames - 1) * VIDEO_SCALE_FACTORS.time + 1

        pixel_shape = VideoPixelShape(
            batch=1, frames=pixel_f, height=pixel_h, width=pixel_w, fps=val_cfg.frame_rate,
        )

        # Run restoration denoising
        sampling_ctx = progress.start_sampling(num_prompts=1, num_steps=val_cfg.inference_steps)
        sampling_ctx.start_video(0)

        x0_model = X0Model(trainer._transformer)
        cfg_guider = CFGGuider(val_cfg.guidance_scale)
        stg_guider = STGGuider(getattr(val_cfg, "stg_scale", 0.0))
        stg_ptb_config = _build_stg_perturbation_config(val_cfg) if stg_guider.enabled() else None

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            video_out, audio_out = _run_restoration(
                lq_video_cond=lq_video_cond,
                lq_audio_tokens=lq_audio_tokens,
                pixel_shape=pixel_shape,
                data_idx=0,
                device=device,
                x0_model=x0_model,
                cfg_guider=cfg_guider,
                stg_guider=stg_guider,
                stg_ptb_config=stg_ptb_config,
                val_cfg=val_cfg,
                sampling_ctx=sampling_ctx,
                text_embeddings=text_embeddings,
            )

            logger.info(
                f"  [{mode_tag}] video_out stats: min={video_out.min().item():.4f} "
                f"max={video_out.max().item():.4f} mean={video_out.mean().item():.4f} "
                f"shape={list(video_out.shape)}"
            )

            # Decode HQ latent → GT pixels, LQ latent → LQ pixels
            _td = getattr(trainer, "_tiny_decoder", None)
            if _td is not None:
                # TinyDecoder: [B, C, F, H, W] → permute → [B, F, C, H, W] → decode → [B, F, 3, H_px, W_px]
                hq_ntchw = hq_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
                lq_ntchw = lq_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
                gt_01 = _td.decode_video(hq_ntchw, parallel=True, show_progress_bar=False)
                lq_01 = _td.decode_video(lq_ntchw, parallel=True, show_progress_bar=False)
                # [B, T, 3, H, W] → [B, 3, T, H, W], scale [0,1] → [-1,1]
                gt_pixel = gt_01.permute(0, 2, 1, 3, 4) * 2 - 1
                lq_pixel = lq_01.permute(0, 2, 1, 3, 4) * 2 - 1
            else:
                trainer._vae_decoder.to(device)
                gt_pixel = torch.cat(
                    list(trainer._vae_decoder.tiled_decode(hq_latent, _TC.default())), dim=2
                )
                lq_pixel = torch.cat(
                    list(trainer._vae_decoder.tiled_decode(lq_latent, _TC.default())), dim=2
                )
                trainer._vae_decoder.to("cpu")

            # GT/LQ audio: prefer raw waveform (skip VAE round-trip)
            gt_audio_wav = None
            lq_audio_wav = None
            raw_audio_sr = batch.get("raw_audio_sr")
            if generate_audio and "raw_hq_audio" in batch and batch["raw_hq_audio"] is not None:
                gt_audio_wav = batch["raw_hq_audio"].float().cpu()
                lq_audio_wav = batch["raw_lq_audio"].float().cpu() if "raw_lq_audio" in batch else None
            elif generate_audio and trainer._audio_vae is not None and trainer._vocoder is not None:
                trainer._audio_vae.to(device)
                trainer._vocoder.to(device)
                if "audio_hq_latents" in batch:
                    ahq = batch["audio_hq_latents"]["latents"][0:1].to(device=device, dtype=torch.bfloat16)
                    gt_audio_wav = trainer._vocoder(trainer._audio_vae(ahq)).squeeze(0).float()
                if "audio_lq_latents" in batch:
                    alq_raw = batch["audio_lq_latents"]["latents"][0:1].to(device=device, dtype=torch.bfloat16)
                    lq_audio_wav = trainer._vocoder(trainer._audio_vae(alq_raw)).squeeze(0).float()
                trainer._vocoder.to("cpu")
                trainer._audio_vae.to("cpu")

        sampling_ctx.cleanup()

        results: list[tuple[int, Path]] = []
        if not is_main:
            return results

        from torchvision.utils import save_image as tv_save_image

        output_dir = Path(trainer._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        step_str = f"step_{trainer._global_step:06d}"
        prefix = f"[{mode_tag}]_{step_str}"

        lq_pixel_01 = lq_pixel.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5
        gt_pixel_01 = gt_pixel.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5
        restored_pixel_01 = video_out.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5

        # Audio sample rate for saving
        _voc_sr = trainer._vocoder.output_sampling_rate if trainer._vocoder is not None else None
        _raw_sr = raw_audio_sr if raw_audio_sr is not None else _voc_sr

        def _trim_audio(wav, sr):
            if wav is None:
                return None, None
            # Ensure (2, T) stereo format for _write_audio compatibility
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] == 1:
                wav = wav.expand(2, -1).contiguous()
            if wav.shape[-1] % 2 != 0:
                wav = wav[..., :-1]
            return wav, sr

        lq_aud, lq_aud_sr = _trim_audio(lq_audio_wav, _raw_sr)
        gt_aud, gt_aud_sr = _trim_audio(gt_audio_wav, _raw_sr)
        res_aud, res_aud_sr = _trim_audio(audio_out, _voc_sr)
        # No generated audio → reuse LQ audio for restored video
        if res_aud is None and lq_aud is not None:
            res_aud, res_aud_sr = lq_aud, lq_aud_sr

        # Save LQ video + audio
        save_video(
            video_tensor=lq_pixel_01, output_path=output_dir / f"{prefix}_lq.mp4",
            fps=val_cfg.frame_rate, audio=lq_aud, audio_sample_rate=lq_aud_sr,
        )

        # Save restored video + audio
        save_video(
            video_tensor=restored_pixel_01, output_path=output_dir / f"{prefix}_restored.mp4",
            fps=val_cfg.frame_rate, audio=res_aud, audio_sample_rate=res_aud_sr,
        )

        # Save GT video + audio
        save_video(
            video_tensor=gt_pixel_01, output_path=output_dir / f"{prefix}_gt.mp4",
            fps=val_cfg.frame_rate, audio=gt_aud, audio_sample_rate=gt_aud_sr,
        )

        # Mid-frame 3-panel comparison: LQ | Restored | GT
        mid = pixel_f // 2
        lq_frame = lq_pixel_01[:, mid]
        restored_frame = restored_pixel_01[:, mid]
        gt_frame = gt_pixel_01[:, mid]
        # Resize LQ if resolution differs (LQ latent may be lower res)
        if lq_frame.shape != gt_frame.shape:
            from torchvision.transforms.functional import resize as tv_resize
            lq_frame = tv_resize(lq_frame, list(gt_frame.shape[-2:]), antialias=True)
        compare = torch.cat([lq_frame, restored_frame, gt_frame], dim=2)
        tv_save_image(compare, str(output_dir / f"{prefix}_compare.png"))

        results.append((0, output_dir / f"{prefix}_restored.mp4"))
        logger.info(f"  [{mode_tag}] step {trainer._global_step}: saved lq/restored/gt/compare")

        import json as _json
        train_caption = batch.get("caption", "")
        (output_dir / f"{prefix}_caption.json").write_text(
            _json.dumps({"caption": train_caption}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return results

    @torch.no_grad()
    def _restoration_sample_videos(progress):
        """Alternating validation: [VAL] random samples ↔ [TRAIN] current batch."""
        from accelerate import DistributedType

        val_cfg = trainer._config.validation
        generate_audio = val_cfg.generate_audio

        trainer._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        rank = trainer._accelerator.process_index
        world_size = trainer._accelerator.num_processes
        device = trainer._accelerator.device
        is_main = trainer._accelerator.is_main_process

        # --- Determine mode: alternating [VAL] ↔ [TRAIN] ---
        is_val_mode = (_val_state["toggle"] == 0)
        _val_state["toggle"] = 1 - _val_state["toggle"]

        if not is_val_mode:
            # [TRAIN] mode: use current training batch
            return _train_mode_validation(
                progress, val_cfg, generate_audio, device, is_main, rank, world_size,
            )

        # [VAL] mode: random samples from val_dataset
        mode_tag = "VAL"
        n = min(num_val_samples, len(val_dataset))
        if n == 0:
            logger.warning("Validation dataset is empty — skipping validation")
            return []

        rng = random.Random(val_cfg.seed + trainer._global_step)
        all_indices = rng.sample(range(len(val_dataset)), n)

        logger.info(
            f"Validation [{mode_tag}] step {trainer._global_step}: "
            f"{n} sample(s), indices={all_indices[:5]}{'...' if n > 5 else ''}"
        )

        # Split work across ranks (round-robin)
        rank_indices = [all_indices[i] for i in range(rank, len(all_indices), world_size)]

        # FSDP: every rank must run the same number of forwards
        work: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if trainer._accelerator.distributed_type == DistributedType.FSDP and world_size > 1:
            max_per_rank = math.ceil(len(all_indices) / world_size)
            pad_idx = rank_indices[-1] if rank_indices else all_indices[0]
            work += [(pad_idx, False)] * (max_per_rank - len(work))

        sampling_ctx = progress.start_sampling(
            num_prompts=len(work),
            num_steps=val_cfg.inference_steps,
        )

        output_dir = Path(trainer._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        results: list[tuple[int, Path]] = []
        x0_model = X0Model(trainer._transformer)
        cfg_guider = CFGGuider(val_cfg.guidance_scale)
        stg_guider = STGGuider(getattr(val_cfg, "stg_scale", 0.0))
        stg_ptb_config = _build_stg_perturbation_config(val_cfg) if stg_guider.enabled() else None

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            for local_i, (data_idx, save_output) in enumerate(work):
                sampling_ctx.start_video(local_i)

                sample = val_dataset[data_idx]
                video_pixel = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                caption = sample["caption"]
                slug = _caption_to_slug(caption)
                lq_audio_sr = int(sample.get("audio_sample_rate", 44100))

                # Encode LQ video (VAL mode: encode raw input — no degradation)
                from ltx_core.model.video_vae import TilingConfig as _TC
                uuid = sample["sample_id"]

                trainer._vae_encoder.to(device)
                lq_video_latent = trainer._vae_encoder.tiled_encode(video_pixel, tiling_config=_TC.default())
                trainer._vae_encoder.to("cpu")
                # Cross-res SR: pass 5D latent; same-res: patchify to 3D
                hq_latent_h = height // VIDEO_SCALE_FACTORS.height
                hq_latent_w = width // VIDEO_SCALE_FACTORS.width
                if lq_video_latent.shape[-2:] != (hq_latent_h, hq_latent_w):
                    lq_video_cond = lq_video_latent
                else:
                    lq_video_cond = video_patchifier.patchify(lq_video_latent)
                lq_pixel = video_pixel

                # Encode LQ audio (VAL: encode raw)
                lq_audio_tokens, lq_audio_waveform, lq_audio_sr = _encode_lq_audio(
                    sample, device, generate_audio
                )

                # Build VideoPixelShape for tools
                pixel_shape = VideoPixelShape(
                    batch=1, frames=num_frames, height=height, width=width, fps=val_cfg.frame_rate,
                )

                # Load per-sample text embeddings (matches the caption used in training)
                sample_text_emb = _load_sample_text_embeddings(uuid, device)

                # Restoration denoising
                video_out, audio_out = _run_restoration(
                    lq_video_cond=lq_video_cond,
                    lq_audio_tokens=lq_audio_tokens,
                    pixel_shape=pixel_shape,
                    data_idx=data_idx,
                    device=device,
                    x0_model=x0_model,
                    cfg_guider=cfg_guider,
                    stg_guider=stg_guider,
                    stg_ptb_config=stg_ptb_config,
                    val_cfg=val_cfg,
                    sampling_ctx=sampling_ctx,
                    text_embeddings=sample_text_emb,
                )

                logger.info(
                    f"  [{mode_tag}] video_out stats: min={video_out.min().item():.4f} "
                    f"max={video_out.max().item():.4f} mean={video_out.mean().item():.4f} "
                    f"shape={list(video_out.shape)}"
                )

                if not save_output:
                    continue

                # --- Save (rank 0 only) ---
                if is_main:
                    from torchvision.utils import save_image as tv_save_image

                    step_str = f"step_{trainer._global_step:06d}"
                    prefix = f"[{mode_tag}]_{step_str}_{slug}"

                    lq_pixel_01 = (lq_pixel.squeeze(0).float() + 1) / 2
                    hq_pixel_01 = video_out.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5

                    # LQ video + audio
                    lq_audio_for_save = None
                    if lq_audio_waveform is not None:
                        lq_audio_for_save = lq_audio_waveform
                        if lq_audio_for_save.shape[-1] % 2 != 0:
                            lq_audio_for_save = lq_audio_for_save[..., :-1]
                    save_video(
                        video_tensor=lq_pixel_01, output_path=output_dir / f"{prefix}_lq.mp4",
                        fps=val_cfg.frame_rate,
                        audio=lq_audio_for_save,
                        audio_sample_rate=lq_audio_sr if lq_audio_for_save is not None else None,
                    )

                    # Restored video + audio (reuse LQ audio when no generated audio)
                    restored_audio = audio_out
                    restored_audio_sr = trainer._vocoder.output_sampling_rate if restored_audio is not None else None
                    if restored_audio is not None and restored_audio.shape[-1] % 2 != 0:
                        restored_audio = restored_audio[..., :-1]
                    if restored_audio is None and lq_audio_for_save is not None:
                        restored_audio = lq_audio_for_save
                        restored_audio_sr = lq_audio_sr
                    save_video(
                        video_tensor=hq_pixel_01, output_path=output_dir / f"{prefix}_restored.mp4",
                        fps=val_cfg.frame_rate,
                        audio=restored_audio,
                        audio_sample_rate=restored_audio_sr,
                    )

                    # Mid-frame comparison (LQ | Restored)
                    mid = num_frames // 2
                    lq_frame = lq_pixel_01[:, mid]
                    hq_frame = hq_pixel_01[:, mid]
                    if lq_frame.shape != hq_frame.shape:
                        from torchvision.transforms.functional import resize as tv_resize
                        lq_frame = tv_resize(lq_frame, list(hq_frame.shape[-2:]), antialias=True)
                    compare = torch.cat([lq_frame, hq_frame], dim=2)
                    tv_save_image(compare, str(output_dir / f"{prefix}_compare.png"))

                    results.append((data_idx, output_dir / f"{prefix}_restored.mp4"))
                    logger.info(f"  [{mode_tag}] {slug}: saved lq/restored/compare")

                    import json as _json
                    (output_dir / f"{prefix}_caption.json").write_text(
                        _json.dumps({"caption": caption}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        sampling_ctx.cleanup()
        if is_main and results:
            rel = output_dir.relative_to(trainer._config.output_dir)
            logger.info(f"Validation [{mode_tag}] samples saved in {rel}")
        return results

    trainer._sample_videos = _restoration_sample_videos
    logger.info(
        f"Patched validation: {num_val_samples} sample(s) per run, "
        f"alternating [VAL] random ↔ [TRAIN] current batch"
    )


# ---------------------------------------------------------------------------
# Logging Patch: SwanLab + File Logger + Enhanced Console Output
# ---------------------------------------------------------------------------

try:
    import swanlab as _swanlab
except ImportError:
    _swanlab = None


def patch_logging(trainer, raw_config: dict) -> None:
    """Replace W&B with SwanLab, add file logging, and enhance console output.

    Monkey-patches:
      - _init_wandb  → noop (W&B disabled)
      - _log_metrics → SwanLab + file + detailed logger.info per step
    """
    swanlab_cfg = raw_config.get("swanlab", {})
    logging_cfg = raw_config.get("logging", {})
    output_dir = Path(trainer._config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_main = trainer._accelerator.is_main_process

    # ---- File logger (rank 0 only) ----
    file_handler = None
    if is_main and logging_cfg.get("log_file", True):
        log_path = output_dir / "training.log"
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)
        logger.info(f"File logging enabled: {log_path}")

    # ---- SwanLab init (rank 0 only) ----
    swanlab_run = None
    if is_main and swanlab_cfg.get("enabled", False):
        if _swanlab is None:
            logger.warning("swanlab not installed, skipping swanlab logging")
        else:
            api_key = swanlab_cfg.get("api_key")
            if api_key and not os.environ.get("SWANLAB_API_KEY"):
                os.environ["SWANLAB_API_KEY"] = str(api_key)

            exp_name = swanlab_cfg.get("experiment_name")
            if not exp_name:
                exp_name = f"{output_dir.name}_{datetime.now().strftime('%m%d_%H%M')}"

            mode = swanlab_cfg.get("mode", "online" if api_key else "offline")
            swanlab_logdir = str(output_dir / "swanlab")
            os.makedirs(swanlab_logdir, exist_ok=True)

            try:
                swanlab_init_kwargs = {
                    "project": swanlab_cfg.get("project", "LTX23_AV_Restoration"),
                    "experiment_name": exp_name,
                    "config": raw_config,
                    "logdir": swanlab_logdir,
                    "mode": mode,
                }
                if api_key:
                    swanlab_init_kwargs["api_key"] = api_key
                tags = swanlab_cfg.get("tags")
                if tags:
                    swanlab_init_kwargs["tags"] = tags

                swanlab_run = _swanlab.init(**swanlab_init_kwargs)
                logger.info(f"SwanLab initialized: project={swanlab_init_kwargs['project']}, "
                            f"exp={exp_name}, mode={mode}")
            except Exception as exc:
                logger.warning(f"SwanLab init failed: {exc}")

    # ---- Save config snapshot ----
    if is_main:
        config_snapshot = output_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        try:
            import yaml
            with open(config_snapshot, "w") as f:
                yaml.dump(raw_config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Config saved: {config_snapshot}")
        except Exception:
            pass

    # ---- Timing trackers ----
    log_interval = logging_cfg.get("log_interval", 1)
    step_times: deque[float] = deque(maxlen=100)
    _state = {"last_log_time": time.time(), "last_log_step": 0}

    # ---- Per-step JSONL data logger (rank 0 only) ----
    _jsonl_fh = None
    if is_main:
        jsonl_path = output_dir / "step_log.jsonl"
        _jsonl_fh = open(jsonl_path, "a", encoding="utf-8")
        logger.info(f"Per-step JSONL log: {jsonl_path}")

    # ---- GC optimization ----
    gc.disable()
    _gc_interval = 100
    logger.info(f"Python GC disabled; manual gc.collect() every {_gc_interval} steps")

    # ---- Disable W&B ----
    trainer._wandb_run = None

    def _noop_init_wandb(resume_run_id=None):
        trainer._wandb_run = None

    trainer._init_wandb = _noop_init_wandb

    # ---- Per-modality grad norm tracking ----
    # FSDP shards parameters across GPUs — individual param.grad is not directly
    # accessible. We capture the total grad norm from clip_grad_norm_ return value,
    # and rely on train/video_loss + train/audio_loss for per-modality monitoring.
    _grad_norms: dict[str, float] = {}

    _original_clip = trainer._accelerator.clip_grad_norm_

    def _clip_with_grad_norms(params, max_norm, **kwargs):
        total_norm = _original_clip(params, max_norm, **kwargs)
        if total_norm is not None:
            _grad_norms["grad_norm/total"] = float(total_norm)
        return total_norm

    trainer._accelerator.clip_grad_norm_ = _clip_with_grad_norms

    # ---- Enhanced _log_metrics ----
    def _enhanced_log_metrics(metrics: dict[str, float]) -> None:
        if not is_main:
            return

        step = metrics.get("train/global_step", trainer._global_step)
        step_time = metrics.get("train/step_time", 0.0)
        if step_time > 0:
            step_times.append(step_time)

        # Merge grad norms into metrics
        if _grad_norms:
            metrics.update(_grad_norms)

        # Merge per-modality loss from strategy
        strategy = trainer._training_strategy
        v_loss = getattr(strategy, "_last_video_loss", None)
        a_loss = getattr(strategy, "_last_audio_loss", None)
        px_loss = getattr(strategy, "_last_pixel_loss", None)
        lpips_loss = getattr(strategy, "_last_lpips_loss", None)
        wavelet_loss = getattr(strategy, "_last_wavelet_loss", None)
        ae_loss = getattr(strategy, "_last_audio_e2e_loss", None)
        stft_loss = getattr(strategy, "_last_audio_stft_loss", None)
        if v_loss is not None:
            metrics["train/video_loss"] = v_loss
        if a_loss is not None:
            metrics["train/audio_loss"] = a_loss
        if px_loss is not None and px_loss > 0:
            metrics["train/pixel_loss"] = px_loss
        if lpips_loss is not None and lpips_loss > 0:
            metrics["train/lpips_loss"] = lpips_loss
        if wavelet_loss is not None and wavelet_loss > 0:
            metrics["train/wavelet_loss"] = wavelet_loss
        if ae_loss is not None and ae_loss > 0:
            metrics["train/audio_e2e_loss"] = ae_loss
        if stft_loss is not None and stft_loss > 0:
            metrics["train/audio_stft_loss"] = stft_loss

        # Per-step JSONL log
        if _jsonl_fh is not None:
            record = {
                "step": int(step),
                "loss": metrics.get("train/loss", 0.0),
                "video_loss": metrics.get("train/video_loss", 0.0),
                "audio_loss": metrics.get("train/audio_loss", 0.0),
                "lr": metrics.get("train/learning_rate", 0.0),
                "step_time": round(step_time, 3),
                "timestamp": time.time(),
            }
            for k, v in metrics.items():
                if "sigma" in k.lower():
                    record[k.split("/")[-1]] = round(float(v), 6)
            for k, v in _grad_norms.items():
                record[k.replace("/", "_")] = round(float(v), 6)
            nan_count = getattr(trainer, "_nan_count", 0)
            if nan_count > 0:
                record["nan_count"] = nan_count
            _jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if int(step) % 10 == 0:
                _jsonl_fh.flush()

        # Periodic GC
        if int(step) % _gc_interval == 0:
            gc.collect()

        # SwanLab logging (every step)
        if swanlab_run is not None:
            flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            try:
                _swanlab.log(flat, step=int(step))
            except Exception:
                pass

        # Console + file logging (every log_interval steps)
        if log_interval > 0 and int(step) % log_interval == 0:
            loss = metrics.get("train/loss", 0.0)
            lr = metrics.get("train/learning_rate", 0.0)
            total_steps = trainer._config.optimization.steps

            avg_time = sum(step_times) / len(step_times) if step_times else 0.0
            remaining = (total_steps - int(step)) * avg_time
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)

            # GPU memory
            try:
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                mem_str = f"GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
            except Exception:
                mem_str = ""

            # Sigma bucket info if available
            sigma_info = ""
            for k, v in metrics.items():
                if "sigma" in k.lower() and "loss" in k.lower():
                    sigma_info += f" {k.split('/')[-1]}={v:.4f}"

            # Grad norm info
            gn_total = _grad_norms.get("grad_norm/total", 0.0)

            # Per-modality loss
            vl = metrics.get("train/video_loss", 0.0)
            al = metrics.get("train/audio_loss", 0.0)
            loss_str = f"v_loss={vl:.5f} a_loss={al:.5f}"

            # Pixel / perceptual / audio E2E loss (distillation only)
            pxl = metrics.get("train/pixel_loss", 0.0)
            lpips_l = metrics.get("train/lpips_loss", 0.0)
            wav_l = metrics.get("train/wavelet_loss", 0.0)
            ael = metrics.get("train/audio_e2e_loss", 0.0)
            stft_l = metrics.get("train/audio_stft_loss", 0.0)
            if pxl > 0 or lpips_l > 0 or ael > 0:
                loss_str += f" px={pxl:.5f}"
                if lpips_l > 0:
                    loss_str += f" lpips={lpips_l:.5f}"
                if wav_l > 0:
                    loss_str += f" wav={wav_l:.5f}"
                loss_str += f" ae={ael:.5f}"
                if stft_l > 0:
                    loss_str += f" stft={stft_l:.5f}"

            logger.info(
                f"Step {int(step):>6d}/{total_steps} | "
                f"loss={loss:.5f} ({loss_str}) | lr={lr:.2e} | "
                f"gn={gn_total:.4f} | "
                f"time={step_time:.2f}s (avg={avg_time:.2f}s) | "
                f"ETA={eta_h}h{eta_m:02d}m | {mem_str}"
                f"{sigma_info}"
            )

    trainer._log_metrics = _enhanced_log_metrics

    # ---- Log start banner ----
    if is_main:
        cfg = trainer._config
        num_procs = trainer._accelerator.num_processes
        local_params = sum(p.numel() for p in trainer._trainable_params)
        total_params = getattr(trainer, "_total_trainable_params", local_params)
        logger.info("=" * 72)
        logger.info("AV Restoration Training")
        logger.info("=" * 72)
        logger.info(f"  Output dir:      {cfg.output_dir}")
        logger.info(f"  Model:           {cfg.model.model_path}")
        logger.info(f"  LoRA structure:  {getattr(trainer, '_lora_structure_from', None) or 'from config'}")
        logger.info(f"  Resume ckpt:     {cfg.model.load_checkpoint or 'none (fresh start)'}")
        logger.info(f"  Training mode:   {cfg.model.training_mode}")
        logger.info(f"  Total steps:     {cfg.optimization.steps}")
        logger.info(f"  Batch size:      {cfg.optimization.batch_size} x {cfg.optimization.gradient_accumulation_steps} accum x {num_procs} GPUs")
        logger.info(f"  Learning rate:   {cfg.optimization.learning_rate}")
        logger.info(f"  Trainable:       {total_params:,} params")
        logger.info(f"  Mixed precision: {cfg.acceleration.mixed_precision_mode}")
        logger.info(f"  SwanLab:         {'enabled' if swanlab_run else 'disabled'}")
        logger.info(f"  Log interval:    every {log_interval} step(s)")
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Cross-Attention Gradient Isolation
# ---------------------------------------------------------------------------

def patch_cross_attention_grad_isolation(trainer, cutoff_layer: int = 24):
    """Register backward hooks on cross-attention modules in early layers to
    block gradient flow between audio and video modalities.

    In layers [0, cutoff_layer): gradients from a2v and v2a cross-attention
    are zeroed, preventing early-layer gradient mixing between modalities.
    In layers [cutoff_layer, 48): gradients flow freely, promoting AV sync.

    This encourages:
      - Early layers: each modality independently learns SR enhancement
      - Late layers: modalities learn temporal/semantic alignment

    Args:
        trainer: LtxvTrainer with transformer already loaded
        cutoff_layer: Layer index where gradient isolation ends (default 24 = halfway)
    """
    base_transformer = (
        trainer._transformer.get_base_model()
        if hasattr(trainer._transformer, "get_base_model")
        else trainer._transformer
    )

    # Access the actual LTXModel inside potential FSDP/LoRA wrappers
    if hasattr(base_transformer, "transformer_blocks"):
        blocks = base_transformer.transformer_blocks
    elif hasattr(base_transformer, "module") and hasattr(base_transformer.module, "transformer_blocks"):
        blocks = base_transformer.module.transformer_blocks
    else:
        logger.warning("Cannot find transformer_blocks for grad isolation — skipping")
        return

    hook_handles = []
    for i, block in enumerate(blocks):
        if i >= cutoff_layer:
            break

        # Zero gradients flowing back through a2v cross-attention output
        if hasattr(block, "audio_to_video_attn"):
            for param in block.audio_to_video_attn.parameters():
                if param.requires_grad:
                    h = param.register_hook(lambda grad: grad * 0.0)
                    hook_handles.append(h)

        # Zero gradients flowing back through v2a cross-attention output
        if hasattr(block, "video_to_audio_attn"):
            for param in block.video_to_audio_attn.parameters():
                if param.requires_grad:
                    h = param.register_hook(lambda grad: grad * 0.0)
                    hook_handles.append(h)

    trainer._grad_isolation_hooks = hook_handles
    logger.info(
        f"Cross-attention gradient isolation: layers 0-{cutoff_layer-1} blocked, "
        f"{cutoff_layer}-47 open ({len(hook_handles)} hooks registered)"
    )


# ---------------------------------------------------------------------------
# Auto-Preprocessing: Raw MP4 → Cached Latents
# ---------------------------------------------------------------------------
# Runs BEFORE trainer creation so the full GPU is available (no FSDP
# transformer loaded yet).  Uses torch.distributed for multi-rank sync;
# accelerate reuses the initialized process group later.
#
# Three-phase loading to keep peak GPU memory low:
#   Phase 1: Video VAE encoder (~2 GB) — encode HQ + degraded LQ videos
#   Phase 2: Text encoder + embeddings processor (~13 GB) — encode captions
#   Phase 3: Audio VAE encoder (~1 GB) — encode HQ + degraded audio
# Between phases the previous model is deleted and CUDA cache emptied.
# ---------------------------------------------------------------------------

def _cache_is_valid(cache_dir: Path, required_dirs: list[str]) -> bool:
    if not cache_dir.is_dir():
        return False
    for d in required_dirs:
        subdir = cache_dir / d
        if not subdir.is_dir() or not any(subdir.glob("*.pt")):
            return False
    return True


def auto_preprocess(raw_config: dict) -> str | None:
    """Auto-preprocess raw AV data on first run (three-phase).

    Must be called BEFORE ``LtxvTrainer()`` so the GPU is free of the
    22 B transformer.  Returns the cache directory path (to be set as
    ``preprocessed_data_root``), or ``None`` if the cache already exists.
    """
    import torch.distributed as dist

    from ltx_av_sr_trainer.degradation.audio_degradation import apply_audio_degradation
    from ltx_av_sr_trainer.degradation.video_degradation import apply_video_degradation
    from ltx_av_sr_trainer.raw_av_dataset import RawAVDataset
    from ltx_trainer.model_loader import (
        load_audio_vae_encoder,
        load_embeddings_processor,
        load_text_encoder,
        load_video_vae_encoder,
    )

    raw_data_root = raw_config.get("data", {}).get("preprocessed_data_root", "")
    cache_dir = Path(raw_data_root) / ".sr_cache"
    strategy_raw = raw_config.get("training_strategy", {})
    with_audio = strategy_raw.get("with_audio", False)
    model_path = raw_config.get("model", {}).get("model_path", "")
    text_encoder_path = raw_config.get("model", {}).get("text_encoder_path", "")
    load_in_8bit = raw_config.get("acceleration", {}).get("load_text_encoder_in_8bit", False)
    seed = raw_config.get("seed", 42)

    required_dirs = ["hq_latents", "lq_latents", "conditions"]
    if with_audio:
        required_dirs += ["audio_hq_latents", "audio_lq_latents"]

    # --- Distributed setup ---
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    def _barrier():
        if dist.is_initialized():
            dist.barrier()

    # --- Check cache ---
    if _cache_is_valid(cache_dir, required_dirs):
        logger.info(f"Using cached preprocessed data: {cache_dir}")
        return str(cache_dir)

    # --- Create output dirs (rank 0 only) ---
    if is_main:
        for d in required_dirs:
            (cache_dir / d).mkdir(parents=True, exist_ok=True)
    _barrier()

    # --- Resolution ---
    pp_res = raw_config.get("data", {}).get("preprocess_resolution")
    if pp_res:
        parts = str(pp_res).split("x")
        width, height, num_frames = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        val_dims = raw_config.get("validation", {}).get("video_dims", [1152, 1920, 121])
        width, height, num_frames = val_dims

    logger.info(
        f"Preprocessing raw data from {raw_data_root} → {cache_dir} "
        f"(rank {rank}/{world_size}, {width}x{height}x{num_frames})"
    )

    # --- Check if Phase 1 (video latents) has partial cache ---
    # If HQ latents exist but conditions/audio are missing, we can skip Phase 1
    # and only run Phase 2/3 for the already-cached samples.
    hq_dir = cache_dir / "hq_latents"
    cached_uuids = sorted({p.stem for p in hq_dir.glob("*.pt")}) if hq_dir.is_dir() else []
    lq_dir = cache_dir / "lq_latents"
    cached_lq_uuids = {p.stem for p in lq_dir.glob("*.pt")} if lq_dir.is_dir() else set()
    cached_uuids = [u for u in cached_uuids if u in cached_lq_uuids]

    phase1_needed = len(cached_uuids) == 0
    if cached_uuids and not phase1_needed:
        logger.info(
            f"[rank {rank}] Found {len(cached_uuids)} cached HQ+LQ latents, "
            f"skipping Phase 1 — only running Phase 2/3 for existing cache"
        )

    from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

    captions: dict[str, str] = {}
    audio_indices: list[int] = []
    dataset = None

    if phase1_needed:
        # --- Full Phase 1: load raw data and encode ---
        dataset = RawAVDataset(
            data_root=raw_data_root,
            target_width=width,
            target_height=height,
            target_frames=num_frames,
            caption_lang="en",
            require_audio=with_audio,
        )
        total = len(dataset)
        my_indices = list(range(rank, total, world_size))
        logger.info(f"Dataset has {total} samples, rank {rank} will process {len(my_indices)}")

        rng = random.Random(seed + rank)

        from ltx_core.model.video_vae import TilingConfig
        enc_tiling = TilingConfig.default()

        logger.info(f"[rank {rank}] Phase 1/3: Loading video VAE encoder...")
        vae_encoder = load_video_vae_encoder(model_path, device=device, dtype=torch.bfloat16)
        vae_encoder.eval()

        with Progress(
            SpinnerColumn(), "[progress.description]{task.description}",
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            disable=(not is_main),
        ) as progress:
            ptask = progress.add_task(f"Phase 1 – Video (rank {rank})", total=len(my_indices))

            for idx in my_indices:
                try:
                    sample = dataset[idx]
                except Exception as e:
                    logger.warning(f"[rank {rank}] Phase 1 skipping idx {idx}: failed to load sample: {e}")
                    progress.advance(ptask)
                    continue

                uuid = sample["sample_id"]

                captions[uuid] = sample.get("caption", "")
                if with_audio and "audio" in sample:
                    audio_indices.append(idx)

                if (cache_dir / "hq_latents" / f"{uuid}.pt").exists():
                    progress.advance(ptask)
                    continue

                try:
                    video = sample["video"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                    fps = sample["fps"]
                    _, _, t, h, w = video.shape

                    with torch.inference_mode():
                        hq_latent = vae_encoder.tiled_encode(video, tiling_config=enc_tiling).squeeze(0).cpu()

                        frames_2d = rearrange(video.float(), "b c t h w -> (b t) c h w")
                        lq_small = F.interpolate(frames_2d, size=(h // 2, w // 2), mode="bicubic", align_corners=False)
                        lq_small = rearrange(lq_small, "(b t) c h w -> b c t h w", b=1, t=t)
                        lq_small = apply_video_degradation(lq_small, rng=rng, downsample_factor=2)
                        lq_full = F.interpolate(
                            rearrange(lq_small, "b c t h w -> (b t) c h w"),
                            size=(h, w), mode="bicubic", align_corners=False,
                        )
                        lq_full = rearrange(lq_full, "(b t) c h w -> b c t h w", b=1, t=t)
                        lq_full = lq_full.clamp(-1, 1).to(device=device, dtype=torch.bfloat16)
                        del lq_small, frames_2d

                        lq_latent = vae_encoder.tiled_encode(
                            lq_full, tiling_config=enc_tiling
                        ).squeeze(0).cpu()
                        del lq_full, video

                    latent_meta = {
                        "num_frames": hq_latent.shape[1],
                        "height": hq_latent.shape[2],
                        "width": hq_latent.shape[3],
                        "fps": fps,
                    }
                    torch.save({"latents": hq_latent, **latent_meta}, cache_dir / "hq_latents" / f"{uuid}.pt")
                    torch.save({"latents": lq_latent, **latent_meta}, cache_dir / "lq_latents" / f"{uuid}.pt")

                except Exception as e:
                    logger.warning(f"[rank {rank}] Phase 1 failed for {uuid}: {e}")

                progress.advance(ptask)

        del vae_encoder
        torch.cuda.empty_cache()
        logger.info(f"[rank {rank}] Phase 1 complete, VAE encoder released")
    else:
        # --- Resume mode: build captions + audio_indices from cached UUIDs ---
        # Only load raw dataset metadata (caption JSON + audio presence check)
        # without loading full video frames. Much faster than full Phase 1.
        dataset = RawAVDataset(
            data_root=raw_data_root,
            target_width=width,
            target_height=height,
            target_frames=num_frames,
            caption_lang="en",
            require_audio=False,
        )
        cached_uuid_set = set(cached_uuids)
        uuid_to_idx: dict[str, int] = {}
        for i, (mp4_path, _) in enumerate(dataset.samples):
            if mp4_path.stem in cached_uuid_set:
                uuid_to_idx[mp4_path.stem] = i

        logger.info(f"[rank {rank}] Building captions for {len(uuid_to_idx)} cached samples...")
        for uuid, idx in uuid_to_idx.items():
            _, caption_path = dataset.samples[idx]
            captions[uuid] = dataset._load_caption(caption_path)
            if with_audio:
                audio_indices.append(idx)

        logger.info(f"[rank {rank}] Phase 1 skipped (using {len(captions)} cached latents)")

    _barrier()

    # Phase 2 runs on rank 0 only (8-bit Gemma + bitsandbytes doesn't work
    # reliably across multiple GPUs loading independently). Text encoding is
    # fast enough on one GPU (~1s per sample).
    all_uuids = sorted(captions.keys())
    my_uuids = all_uuids[rank::world_size]

    # ================================================================
    # Phase 2: Text embeddings  (text encoder ~12 GB + processor ~0.5 GB)
    # Runs on rank 0 only to avoid 8-bit model loading issues on multi-GPU.
    # ================================================================
    phase2_todo = [u for u in all_uuids if not (cache_dir / "conditions" / f"{u}.pt").exists()]
    if phase2_todo and is_main:
        logger.info(f"[rank {rank}] Phase 2/3: Loading text encoder ({len(phase2_todo)} to encode)...")
        text_encoder = load_text_encoder(
            gemma_model_path=text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=False,
        )

        logger.info(f"[rank {rank}] Phase 2/3: Loading embeddings processor...")
        embeddings_processor = load_embeddings_processor(model_path, device=device, dtype=torch.bfloat16)

        with Progress(
            SpinnerColumn(), "[progress.description]{task.description}",
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        ) as progress:
            ptask = progress.add_task(f"Phase 2 – Text (rank {rank})", total=len(phase2_todo))

            for uuid in phase2_todo:
                caption = captions[uuid]
                try:
                    with torch.inference_mode():
                        if caption:
                            hidden_states, prompt_mask = text_encoder.encode(caption, padding_side="left")
                            video_feats, audio_feats = embeddings_processor.feature_extractor(
                                hidden_states, prompt_mask, "left"
                            )
                            cond_data = {
                                "video_prompt_embeds": video_feats[0].cpu().contiguous(),
                                "prompt_attention_mask": prompt_mask[0].cpu().contiguous(),
                            }
                            if audio_feats is not None:
                                cond_data["audio_prompt_embeds"] = audio_feats[0].cpu().contiguous()
                        else:
                            dummy = torch.zeros(256, 4096)
                            cond_data = {
                                "video_prompt_embeds": dummy,
                                "audio_prompt_embeds": dummy[:, :2048],
                                "prompt_attention_mask": torch.zeros(256, dtype=torch.bool),
                            }
                    torch.save(cond_data, cache_dir / "conditions" / f"{uuid}.pt")

                except Exception as e:
                    logger.warning(f"[rank {rank}] Phase 2 failed for {uuid}: {e}")

                progress.advance(ptask)

        del text_encoder, embeddings_processor
        torch.cuda.empty_cache()
        logger.info(f"[rank {rank}] Phase 2 complete, text encoder released")
    elif not phase2_todo:
        logger.info(f"[rank {rank}] Phase 2 skipped (all conditions cached)")
    else:
        logger.info(f"[rank {rank}] Phase 2 waiting (rank 0 encoding text)")

    _barrier()

    # ================================================================
    # Phase 3: Audio latents  (audio VAE encoder ~1 GB)
    # ================================================================
    if with_audio:
        # Partition audio work across ranks
        my_audio_uuids = [u for u in my_uuids if not (cache_dir / "audio_hq_latents" / f"{u}.pt").exists()]

        if my_audio_uuids and dataset is not None:
            from ltx_core.model.audio_vae import encode_audio
            from ltx_core.types import Audio

            logger.info(f"[rank {rank}] Phase 3/3: Loading audio VAE encoder ({len(my_audio_uuids)} samples)...")
            audio_vae_encoder = load_audio_vae_encoder(model_path, device=device, dtype=torch.bfloat16)
            audio_vae_encoder.eval()

            rng_audio = random.Random(seed + rank + 10000)

            # Build uuid→idx mapping for audio loading
            uuid_to_ds_idx = {}
            for i, (mp4_path, _) in enumerate(dataset.samples):
                if mp4_path.stem in set(my_audio_uuids):
                    uuid_to_ds_idx[mp4_path.stem] = i

            with Progress(
                SpinnerColumn(), "[progress.description]{task.description}",
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                disable=(not is_main),
            ) as progress:
                ptask = progress.add_task(f"Phase 3 – Audio (rank {rank})", total=len(my_audio_uuids))

                for uuid in my_audio_uuids:
                    idx = uuid_to_ds_idx.get(uuid)
                    if idx is None:
                        progress.advance(ptask)
                        continue

                    try:
                        sample = dataset[idx]
                    except Exception as e:
                        logger.warning(f"[rank {rank}] Phase 3 skipping {uuid}: failed to load sample: {e}")
                        progress.advance(ptask)
                        continue

                    if "audio" not in sample:
                        progress.advance(ptask)
                        continue

                    try:
                        waveform = sample["audio"].to(device)  # (C, T) or (T,)
                        sr = int(sample.get("audio_sample_rate", 44100))

                        with torch.inference_mode():
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0)  # (1, T)
                            hq_stereo = waveform if waveform.shape[0] >= 2 else waveform.expand(2, -1)
                            hq_audio = Audio(waveform=hq_stereo.unsqueeze(0).to(torch.bfloat16), sampling_rate=sr)
                            hq_audio_latent = encode_audio(hq_audio, audio_vae_encoder).squeeze(0).cpu()

                            lq_waveform = apply_audio_degradation(waveform, rng=rng_audio, sample_rate=sr)
                            if lq_waveform.shape[0] == 1:
                                lq_waveform = lq_waveform.expand(2, -1)
                            lq_audio = Audio(waveform=lq_waveform.unsqueeze(0).to(device=device, dtype=torch.bfloat16), sampling_rate=sr)
                            lq_audio_latent = encode_audio(lq_audio, audio_vae_encoder).squeeze(0).cpu()

                        torch.save({"latents": hq_audio_latent}, cache_dir / "audio_hq_latents" / f"{uuid}.pt")
                        torch.save({"latents": lq_audio_latent}, cache_dir / "audio_lq_latents" / f"{uuid}.pt")

                    except Exception as e:
                        logger.warning(f"[rank {rank}] Phase 3 failed for {uuid}: {e}")

                    progress.advance(ptask)

            del audio_vae_encoder
            torch.cuda.empty_cache()
            logger.info(f"[rank {rank}] Phase 3 complete, audio encoder released")
        else:
            logger.info(f"[rank {rank}] Phase 3 skipped (all audio cached or no dataset)")

    # --- Sync all ranks, return cache path ---
    _barrier()
    logger.info(f"[rank {rank}] All preprocessing phases complete")

    return str(cache_dir)


def _ensure_vae_encoder(trainer) -> None:
    """Ensure trainer has a VAE encoder loaded (needed for validation)."""
    if trainer._vae_encoder is not None:
        return
    from ltx_trainer.model_loader import load_video_vae_encoder

    model_path = trainer._config.model.model_path
    logger.info("Loading video VAE encoder for validation...")
    trainer._vae_encoder = load_video_vae_encoder(model_path, device="cpu", dtype=torch.bfloat16)
    trainer._vae_encoder.eval()
    logger.info("Video VAE encoder loaded (CPU)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    patch_strategy_registry()

    import yaml

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.trainer import LtxvTrainer

    # --- Auto-preprocess raw data BEFORE trainer creation (GPU is empty) ---
    cache_dir = auto_preprocess(raw_config)

    # Save original raw data root for validation (before cache_dir overwrites it)
    raw_data_root_for_val = raw_config.get("data", {}).get("preprocessed_data_root", "")

    # Append timestamp + GPU count to output_dir to prevent overwriting previous runs.
    # Rank 0 writes "timestamp:nonce" to a shared file. All other ranks poll
    # until they see a *new* nonce (different from any stale value already on
    # disk), which guarantees they read this run's timestamp, not a leftover.
    if "output_dir" in raw_config:
        import time as _time

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        ts_file = Path(raw_config["output_dir"]).parent / ".train_timestamp"
        ts_file.parent.mkdir(parents=True, exist_ok=True)

        stale = ts_file.read_text().strip() if ts_file.exists() else ""

        if rank == 0:
            nonce = f"{os.getpid()}_{_time.time_ns()}"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_file.write_text(f"{ts}:{nonce}")
        else:
            ts = None
            for _ in range(600):
                if ts_file.exists():
                    val = ts_file.read_text().strip()
                    if val and val != stale:
                        ts = val.split(":")[0]
                        break
                _time.sleep(0.5)
            if ts is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_config["output_dir"] = f"{raw_config['output_dir']}_{ts}_gpus{num_gpus}"

    # Strip custom fields that upstream LtxTrainerConfig (extra="forbid") doesn't know
    trainer_raw = {k: v for k, v in raw_config.items() if k not in ("swanlab", "logging")}

    # Redirect data root to cache if preprocessing was done
    if cache_dir is not None:
        trainer_raw.setdefault("data", {})["preprocessed_data_root"] = cache_dir

    # Remove custom data fields that DataConfig doesn't accept
    trainer_raw.get("data", {}).pop("preprocess_resolution", None)

    # AVRestorationConfig isn't in upstream's Pydantic discriminated union, so
    # swap in a text_to_video stub for config validation, then restore after.
    av_strategy_raw = trainer_raw.get("training_strategy", {})
    if av_strategy_raw.get("name") == "av_restoration":
        trainer_raw["training_strategy"] = {
            "name": "text_to_video",
            "with_audio": av_strategy_raw.get("with_audio", True),
            "first_frame_conditioning_p": av_strategy_raw.get("first_frame_conditioning_p", 0.0),
        }

    trainer_config = LtxTrainerConfig(**trainer_raw)

    # Restore the real AVRestorationConfig
    if av_strategy_raw.get("name") == "av_restoration":
        from ltx_av_sr_trainer.strategy import AVRestorationConfig
        trainer_config.training_strategy = AVRestorationConfig(**av_strategy_raw)

    SRTrainer.apply(LtxvTrainer)

    trainer = LtxvTrainer(trainer_config)

    patch_logging(trainer, raw_config)
    _ensure_vae_encoder(trainer)

    val_data_root = os.environ.get("VAL_DATA_ROOT", raw_data_root_for_val)
    patch_validation(trainer, val_data_root)

    trainer.train()


if __name__ == "__main__":
    main()
