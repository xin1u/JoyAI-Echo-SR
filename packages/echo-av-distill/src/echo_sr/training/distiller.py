"""EchoDistiller: DMD2 one-step distillation for AV super-resolution.

Distills a multi-step SFT teacher LoRA into a single-step student.
Three LoRA branches (student/teacher/critic) share one 22B base model.

Loss = reg_loss (GT MSE) + dm_loss (DMD score) + pixel_loss + lpips_loss + wavelet_loss
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from echo_sr import logger
from echo_sr.training.trainer import EchoTrainer


class FixedSigmaSampler:
    """Always returns σ=1.0 for single-step distillation."""
    def sample_for(self, latent):
        return torch.ones(latent.shape[0], device=latent.device, dtype=latent.dtype)


class EchoDistiller(EchoTrainer):
    """DMD2 distillation trainer, extends EchoTrainer."""

    def setup(self) -> None:
        """Initialize base + distillation-specific components."""
        cfg = self.config
        distill_cfg = cfg["distillation"]
        model_cfg = cfg["model"]

        # ─── Base model, VAE, prompts, dataset (reuse EchoTrainer) ───
        # But override LoRA setup to use EchoDMD instead of EchoLoRA
        self._setup_base()

        # ─── DMD three-branch LoRA ───
        from echo_sr.model.lora import EchoDMD
        import time as _time
        lora_cfg = cfg["lora"]
        teacher_ckpt = distill_cfg["teacher_checkpoint"]

        # Load teacher state dict once (reused for DMD init + cond_proj)
        from safetensors.torch import load_file
        t0 = _time.time()
        teacher_sd = load_file(teacher_ckpt) if teacher_ckpt.endswith(".safetensors") else torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        if self.is_main:
            logger.info(f"  [timing] Teacher checkpoint loaded in {_time.time() - t0:.1f}s")

        t0 = _time.time()
        self.dmd = EchoDMD(
            self.transformer,
            target_modules=lora_cfg["target_modules"],
            rank=lora_cfg.get("rank", 384),
            alpha=lora_cfg.get("alpha", 384),
            teacher_checkpoint=teacher_ckpt,
            student_init=distill_cfg.get("student_init", "from_teacher"),
            student_checkpoint=model_cfg.get("load_checkpoint") or distill_cfg.get("student_checkpoint"),
            param_dtype=torch.float32,
            teacher_state_dict=teacher_sd,
        )
        if self.is_main:
            logger.info(f"  [timing] EchoDMD initialized in {_time.time() - t0:.1f}s")
        self.lora = self.dmd.student  # for checkpoint saving compatibility

        # Load cond_proj from teacher state dict (already loaded above)
        loaded_cond = 0
        for name, param in self.transformer.named_parameters():
            if "_cond_" in name and "proj" in name:
                key = f"diffusion_model.{name}"
                if key in teacher_sd:
                    param.data.copy_(teacher_sd[key].to(dtype=param.dtype))
                    loaded_cond += 1
        if loaded_cond > 0:
            logger.info(f"Loaded cond_proj: {loaded_cond} tensors from teacher")
        del teacher_sd

        # Count params before FSDP
        self._total_lora_params = self.dmd.student.num_parameters()
        self._total_cond_params = sum(p.numel() for p in self.cond_params)
        self._total_trainable = self._total_lora_params + self._total_cond_params

        # FSDP prepare
        t0 = _time.time()
        self.transformer = self.transformer.to(dtype=torch.float32)
        self.transformer.set_gradient_checkpointing(cfg["optimization"].get("enable_gradient_checkpointing", True))
        self.transformer = self.accelerator.prepare(self.transformer)
        if self.is_main:
            logger.info(f"  [timing] FSDP prepare in {_time.time() - t0:.1f}s")

        # ─── Optimizers ───
        opt_cfg = cfg["optimization"]
        opt_params = cfg.get("optimizer_params", {})

        # Student optimizer
        student_params = list(self.dmd.student.parameters()) + self.cond_params
        self._trainable_params = student_params
        self.optimizer = torch.optim.AdamW(
            student_params,
            lr=float(opt_cfg["learning_rate"]),
            betas=tuple(opt_params.get("betas", [0.0, 0.99])),
            weight_decay=float(opt_params.get("weight_decay", 0.0)),
        )

        # Critic optimizer (uses critic buffers, separate from student params)
        critic_lr = float(distill_cfg.get("critic_lr", 1e-5))
        self.critic_optimizer = torch.optim.AdamW(
            self.dmd.critic_parameters(),
            lr=critic_lr,
            betas=tuple(opt_params.get("betas", [0.0, 0.99])),
            weight_decay=float(opt_params.get("weight_decay", 0.0)),
        )

        # LR Scheduler
        self.scheduler = None
        scheduler_type = opt_cfg.get("scheduler_type")
        if scheduler_type:
            from transformers import get_scheduler
            scheduler_params = opt_cfg.get("scheduler_params", {})
            self.scheduler = get_scheduler(
                scheduler_type, optimizer=self.optimizer,
                num_warmup_steps=scheduler_params.get("warmup_steps", 0),
                num_training_steps=opt_cfg["steps"],
            )
            self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # ─── Perceptual losses ───
        self._pixel_w = float(distill_cfg.get("pixel_loss_weight", 0.0))
        self._lpips_w = float(distill_cfg.get("lpips_loss_weight", 0.0))
        self._wavelet_w = float(distill_cfg.get("wavelet_loss_weight", 0.0))
        self._temporal_w = float(distill_cfg.get("temporal_loss_weight", 0.0))
        self._pixel_temporal_w = float(distill_cfg.get("pixel_temporal_loss_weight", 0.0))
        self._latent_freq_w = float(distill_cfg.get("latent_freq_loss_weight", 0.0))
        self._num_decode_frames = int(distill_cfg.get("num_decode_frames", 8))
        self._dm_w = float(distill_cfg.get("dm_loss_weight", 1.0))
        self._reg_w = float(distill_cfg.get("reg_loss_weight", 1.0))
        self._audio_e2e_w = float(distill_cfg.get("audio_e2e_loss_weight", 0.0))
        self._audio_stft_w = float(distill_cfg.get("audio_stft_loss_weight", 0.0))
        self._critic_interval = int(distill_cfg.get("critic_update_interval", 1))
        self._enable_dmd = bool(distill_cfg.get("enable_dmd", True))

        self._lpips_model = None
        if self._lpips_w > 0:
            from echo_sr.training.perceptual import load_lpips_model
            self._lpips_model = load_lpips_model(self.device)
            logger.info("LPIPS model loaded (VGG)")

        # TinyDecoder for pixel-space losses (fallback)
        self._tiny_decoder = None
        self._train_tiny_decoder = bool(distill_cfg.get("train_tiny_decoder", False))
        self._full_decoder = None
        self._use_full_decoder = bool(distill_cfg.get("use_full_decoder", False))

        if self._pixel_w > 0 or self._lpips_w > 0 or self._wavelet_w > 0:
            if self._use_full_decoder:
                from echo_sr.model.loader import load_video_vae_decoder
                self._full_decoder = load_video_vae_decoder(
                    cfg["model"]["model_path"], self.device, dtype=torch.bfloat16,
                )
                logger.info("Full VAE decoder loaded for pixel loss (frozen, 1-frame decode)")
            else:
                from echo_sr.validation.tiny_decoder import TAEHV
                td_path = cfg.get("validation", {}).get("tiny_decoder_path", "ckpt/tinydecoder/taeltx2_3_wide.pth")
                self._tiny_decoder = TAEHV(checkpoint_path=td_path).to(self.device, torch.bfloat16)
                if self._train_tiny_decoder:
                    self._tiny_decoder.train().requires_grad_(True)
                    all_params = student_params + list(self._tiny_decoder.parameters())
                    self.optimizer = torch.optim.AdamW(
                        all_params, lr=float(opt_cfg["learning_rate"]),
                        betas=tuple(opt_params.get("betas", [0.0, 0.99])),
                        weight_decay=float(opt_params.get("weight_decay", 0.0)),
                    )
                    self.optimizer = self.accelerator.prepare(self.optimizer)
                else:
                    self._tiny_decoder.eval().requires_grad_(False)
                logger.info(f"TinyDecoder: {'TRAINABLE' if self._train_tiny_decoder else 'FROZEN'}")

        # Fixed sigma sampler
        self._fixed_sampler = FixedSigmaSampler()

        # Finish setup (output dir, logging, banner)
        self._setup_output_and_logging()

        logger.info(f"  DM loss: {self._dm_w} ({'active' if self._enable_dmd else 'off'})")
        logger.info(f"  Reg loss: {self._reg_w}")
        logger.info(f"  Pixel loss: {self._pixel_w}")
        logger.info(f"  LPIPS loss: {self._lpips_w}")
        logger.info(f"  Wavelet loss: {self._wavelet_w}")
        logger.info(f"  Temporal loss: {self._temporal_w} (latent-space)")
        logger.info(f"  Pixel temporal loss: {self._pixel_temporal_w} (pixel-space)")
        logger.info(f"  Latent freq loss: {self._latent_freq_w} (uniform 3D FFT)")
        logger.info(f"  Critic interval: {self._critic_interval}")

    def _setup_base(self) -> None:
        """Initialize accelerator, model, cond_proj, VAE, prompts, strategy, dataset, dataloader."""
        cfg = self.config
        import time as _time
        from accelerate import Accelerator

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg["optimization"].get("gradient_accumulation_steps", 1),
            mixed_precision="bf16" if cfg["acceleration"].get("mixed_precision_mode") == "bf16" else "no",
        )
        self.device = self.accelerator.device
        self.is_main = self.accelerator.is_main_process
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes

        from echo_sr.model.loader import (
            load_transformer, load_vae_encoder,
            init_cond_proj, encode_fixed_prompts,
        )

        model_cfg = cfg["model"]

        # Load only transformer (skip decoder/vocoder — loaded lazily at validation)
        t0 = _time.time()
        self.transformer = load_transformer(model_cfg["model_path"], dtype=torch.bfloat16)
        if self.is_main:
            logger.info(f"  [timing] Transformer loaded in {_time.time() - t0:.1f}s")

        # Cond proj
        data_cfg = cfg["data"]
        from ltx_core.types import SpatioTemporalScaleFactors
        scale = SpatioTemporalScaleFactors.default()
        hq_w, hq_h, _ = data_cfg["hq_resolution"]
        lq_w, lq_h, _ = data_cfg["lq_resolution"]
        sr_spatial = None
        cond_proj_cfg = cfg.get("cond_proj")
        if cond_proj_cfg and cond_proj_cfg.get("type") == "latent_2x":
            pass
        elif lq_w != hq_w or lq_h != hq_h:
            sr_spatial = {
                "lq_h": lq_h // scale.height, "lq_w": lq_w // scale.width,
                "hq_h": hq_h // scale.height, "hq_w": hq_w // scale.width,
            }
        self.cond_params = init_cond_proj(self.transformer, sr_spatial, cond_proj_cfg)

        # VAE encoder (needed for online/tar data)
        t0 = _time.time()
        self.vae_encoder = load_vae_encoder(model_cfg["model_path"], self.device)
        if self.is_main:
            logger.info(f"  [timing] VAE encoder loaded in {_time.time() - t0:.1f}s")

        # Audio VAE encoder — skip in pure video mode
        self.audio_vae_encoder = None
        if cfg["training_strategy"].get("with_audio", True):
            from echo_sr.model.loader import load_audio_vae_encoder
            t0 = _time.time()
            self.audio_vae_encoder = load_audio_vae_encoder(model_cfg["model_path"], self.device)
            if self.is_main:
                logger.info(f"  [timing] Audio VAE encoder loaded in {_time.time() - t0:.1f}s")

        # Prompts (uses cache if available — fast)
        from echo_sr.training.prompts import SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT
        cache_path = Path(cfg.get("prompt_cache_path", "ckpt/prompt/sr_prompt_embeddings.pt"))
        self.cond_feats, self.val_embeds = encode_fixed_prompts(
            model_cfg["model_path"], model_cfg["text_encoder_path"], self.device,
            SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT, cache_path,
        )

        # Strategy
        from echo_sr.training.strategy import AVRestorationConfig, AVRestorationStrategy
        strategy_cfg = cfg["training_strategy"]
        self.strategy = AVRestorationStrategy(AVRestorationConfig(**strategy_cfg))

        # Timestep sampler (for SFT-style reg loss)
        from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler
        self.timestep_sampler = ShiftedLogitNormalTimestepSampler()

        # Dataset + dataloader
        from torch.utils.data import DataLoader, DistributedSampler

        opt_cfg = cfg["optimization"]
        max_f = data_cfg["hq_resolution"][2]
        dataset_type = data_cfg.get("dataset_type", "online")

        if dataset_type == "tar":
            from echo_sr.data.tar_dataset import TarVideoDataset, tar_collate_fn
            self.dataset = TarVideoDataset(
                data_index_files=data_cfg["data_index_files"],
                hq_width=hq_w, hq_height=hq_h,
                lq_width=lq_w, lq_height=lq_h,
                target_frames=max_f,
                fps=25,
                seed=cfg.get("seed", 42),
            )
            # IterableDataset — no sampler needed (shards internally)
            self.dataloader = DataLoader(
                self.dataset, batch_size=opt_cfg.get("batch_size", 1),
                num_workers=data_cfg.get("num_dataloader_workers", 1),
                collate_fn=tar_collate_fn,
                pin_memory=True,
            )
        else:
            from echo_sr.data.online_dataset import OnlineAVDataset
            from echo_sr.data.collate import online_collate_fn
            self.dataset = OnlineAVDataset(
                data_root=data_cfg["preprocessed_data_root"],
                hq_width=hq_w, hq_height=hq_h, lq_width=lq_w, lq_height=lq_h,
                target_frames=max_f,
                with_audio=strategy_cfg.get("with_audio", True),
                seed=cfg.get("seed", 42),
            )
            sampler = DistributedSampler(
                self.dataset, num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index, shuffle=True, drop_last=True,
            ) if self.accelerator.num_processes > 1 else None
            self.dataloader = DataLoader(
                self.dataset, batch_size=opt_cfg.get("batch_size", 1),
                shuffle=(sampler is None), sampler=sampler, drop_last=True,
                num_workers=data_cfg.get("num_dataloader_workers", 0),
                collate_fn=online_collate_fn,
            )

    def _setup_output_and_logging(self) -> None:
        """Output dir, logging, banner — reuse EchoTrainer methods."""
        cfg = self.config
        import os
        from datetime import datetime

        output_dir = cfg.get("output_dir", "outputs/distill")
        num_gpus = self.world_size

        if self.is_main:
            ts = datetime.now().strftime("%Y%m%d_%H%M00")
        else:
            ts = None
        if num_gpus > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                if self.is_main:
                    ts_tensor = torch.tensor([ord(c) for c in ts], dtype=torch.uint8, device=self.device)
                else:
                    ts_tensor = torch.zeros(15, dtype=torch.uint8, device=self.device)
                dist.broadcast(ts_tensor, src=0)
                ts = "".join(chr(c) for c in ts_tensor.cpu().tolist())
        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M00")
        self.output_dir = Path(f"{output_dir}_{ts}_gpus{num_gpus}")
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self._setup_logging()
        self._log_start_banner()

    def _log_start_banner(self) -> None:
        """Override to handle TarVideoDataset (no __len__, no preprocessed_data_root)."""
        if not self.is_main:
            return

        cfg = self.config
        opt_cfg = cfg["optimization"]
        data_cfg = cfg["data"]
        lora_cfg = cfg["lora"]
        distill_cfg = cfg["distillation"]
        total_params = self._total_trainable
        lora_params = self._total_lora_params
        cond_params = self._total_cond_params

        dataset_type = data_cfg.get("dataset_type", "preprocessed")
        if dataset_type == "tar":
            dataset_info = f"{len(self.dataset.tar_paths)} tar files"
        else:
            dataset_info = f"{len(self.dataset)} samples from {data_cfg.get('preprocessed_data_root', 'N/A')}"

        logger.info("=" * 72)
        logger.info("Echo SR DMD2 Distillation — Video Super-Resolution")
        logger.info("=" * 72)
        logger.info(f"  Output dir:        {self.output_dir}")
        logger.info(f"  Teacher ckpt:      {distill_cfg['teacher_checkpoint']}")
        logger.info(f"  Student init:      {distill_cfg.get('student_init', 'from_teacher')}")
        logger.info(f"  Student ckpt:      {cfg['model'].get('load_checkpoint') or 'none'}")
        logger.info(f"  Resolution:        LQ {data_cfg['lq_resolution']} → HQ {data_cfg['hq_resolution']}")
        logger.info(f"  Dataset:           {dataset_info}")
        logger.info(f"  ──── Training ────")
        logger.info(f"  Total steps:       {opt_cfg['steps']}")
        logger.info(f"  Batch size:        {opt_cfg.get('batch_size', 1)} x {self.world_size} GPUs")
        logger.info(f"  Learning rate:     {opt_cfg['learning_rate']}")
        logger.info(f"  Critic LR:         {distill_cfg.get('critic_lr', 1e-5)}")
        logger.info(f"  ──── Model ────")
        logger.info(f"  LoRA rank:         {lora_cfg.get('rank', 384)}")
        logger.info(f"  LoRA modules:      {len(self.lora._targets)}")
        logger.info(f"  LoRA params:       {lora_params:,}")
        logger.info(f"  Cond proj params:  {cond_params:,}")
        logger.info(f"  Total trainable:   {total_params:,}")
        logger.info(f"  ──── Loss weights ────")
        logger.info(f"  DM:     {distill_cfg.get('dm_loss_weight', 1.0)}")
        logger.info(f"  Reg:    {distill_cfg.get('reg_loss_weight', 0.1)}")
        logger.info(f"  Pixel:  {distill_cfg.get('pixel_loss_weight', 2.0)}")
        logger.info(f"  LPIPS:  {distill_cfg.get('lpips_loss_weight', 2.0)}")
        logger.info(f"  Wavelet:{distill_cfg.get('wavelet_loss_weight', 5.0)}")
        logger.info("=" * 72)

    def _train_step(self, batch: dict) -> torch.Tensor:
        """DMD distillation step."""
        device = self.device
        from ltx_core.model.video_vae import TilingConfig

        # ─── VAE encode (same as EchoTrainer) ───
        hq_video = batch["hq_video"].to(device=device, dtype=torch.bfloat16)
        lq_video = batch["lq_video"].to(device=device, dtype=torch.bfloat16)
        if hq_video.dim() == 4:
            hq_video = hq_video.unsqueeze(0)
            lq_video = lq_video.unsqueeze(0)

        from echo_sr.data.video_degradation import apply_video_degradation
        with torch.no_grad():
            rng = random.Random(42 + self.global_step * self.world_size + self.rank)
            lq_video = apply_video_degradation(lq_video.float(), rng=rng, downsample_factor=4).to(torch.bfloat16)

        enc_tiling = TilingConfig.default()
        with torch.no_grad():
            hq_latent = self.vae_encoder.tiled_encode(hq_video, tiling_config=enc_tiling)
            lq_latent = self.vae_encoder.tiled_encode(lq_video, tiling_config=enc_tiling)

        b = hq_latent.shape[0]
        latent_meta = {
            "num_frames": torch.tensor([hq_latent.shape[2]] * b),
            "height": torch.tensor([hq_latent.shape[3]] * b),
            "width": torch.tensor([hq_latent.shape[4]] * b),
            "fps": batch["fps"] if isinstance(batch["fps"], torch.Tensor) else torch.tensor([batch["fps"]] * b),
        }

        # Audio
        hq_audio_lat = lq_audio_lat = None
        if self.audio_vae_encoder is not None:
            hq_audio = batch.get("hq_audio")
            lq_audio = batch.get("lq_audio")
            if hq_audio is not None and not (hq_audio == 0).all():
                audio_sr = batch["audio_sr"]
                if isinstance(audio_sr, torch.Tensor):
                    audio_sr = int(audio_sr[0].item())
                with torch.no_grad():
                    hq_audio_lat = self._encode_audio(hq_audio, audio_sr)
                    lq_audio_lat = self._encode_audio(lq_audio, audio_sr)

        # Text conditions
        feats = self.cond_feats
        conditions = {
            "video_prompt_embeds": feats["video_embeds"][0:1].expand(b, -1, -1).to(device),
            "prompt_attention_mask": feats["attention_mask"][0:1].expand(b, -1).to(device),
        }
        if feats["audio_embeds"] is not None:
            conditions["audio_prompt_embeds"] = feats["audio_embeds"][0:1].expand(b, -1, -1).to(device)

        # Assemble
        assembled = {
            "hq_latents": {"latents": hq_latent, **latent_meta},
            "lq_latents": {"latents": lq_latent, **latent_meta},
            "conditions": conditions,
        }
        if self.audio_vae_encoder is not None:
            if hq_audio_lat is not None:
                assembled["audio_hq_latents"] = {"latents": hq_audio_lat}
                assembled["audio_lq_latents"] = {"latents": lq_audio_lat}
            else:
                dummy = torch.zeros(b, 8, 1, 16, device=device, dtype=torch.bfloat16)
                assembled["audio_hq_latents"] = {"latents": dummy}
                assembled["audio_lq_latents"] = {"latents": dummy}

        self._last_batch = assembled
        self._last_batch_raw_audio = {
            "hq_audio": batch.get("hq_audio"), "lq_audio": batch.get("lq_audio"),
            "audio_sr": batch.get("audio_sr"),
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Student forward (σ=1.0) → GT regression loss
        # ═══════════════════════════════════════════════════════════════
        self.dmd.activate("student")
        model_inputs = self.strategy.prepare_training_inputs(assembled, self._fixed_sampler)
        video_pred, audio_pred = self.transformer(
            video=model_inputs.video, audio=model_inputs.audio, perturbations=None,
        )
        loss_reg = self.strategy.compute_loss(video_pred, audio_pred, model_inputs)

        # x0_student = noisy - v * σ (σ=1 → x0 = noise - v)
        x0_student = model_inputs.video.latent.detach() - video_pred.detach()

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: DMD loss
        # ═══════════════════════════════════════════════════════════════
        loss_dm = torch.zeros_like(loss_reg)
        if self._enable_dmd and self._dm_w > 0:
            from dataclasses import replace

            sigma_dm = torch.rand(b, device=device) * 0.8 + 0.1
            noise_dm = torch.randn_like(x0_student)
            sigma_exp = sigma_dm.view(-1, 1, 1)
            noisy_dm = (1 - sigma_exp) * x0_student + sigma_exp * noise_dm

            dm_video_mod = replace(
                model_inputs.video,
                latent=noisy_dm,
                sigma=sigma_dm,
                timesteps=sigma_dm.view(-1, 1).expand(-1, noisy_dm.shape[1]),
            )

            # Teacher score
            self.dmd.activate("teacher")
            with torch.no_grad():
                pred_real, _ = self.transformer(video=dm_video_mod, audio=None, perturbations=None)
                x0_real = noisy_dm - pred_real * sigma_exp

            # Critic score
            self.dmd.activate("critic")
            with torch.no_grad():
                pred_fake, _ = self.transformer(video=dm_video_mod, audio=None, perturbations=None)
                x0_fake = noisy_dm - pred_fake * sigma_exp

            # Back to student
            self.dmd.activate("student")

            dm_update = (x0_real - x0_fake).float()
            normalizer = dm_update.abs().mean(dim=[1, 2], keepdim=True).clamp(min=1e-6)
            dm_update = torch.nan_to_num(dm_update / normalizer)

            x0_with_grad = model_inputs.video.latent - video_pred
            loss_dm = 0.5 * F.mse_loss(
                x0_with_grad.float(),
                (x0_with_grad.float() + dm_update).detach(),
            ).unsqueeze(0)

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Pixel / LPIPS / Wavelet loss
        # ═══════════════════════════════════════════════════════════════
        loss_pixel = torch.zeros(1, device=device)
        loss_lpips = torch.zeros(1, device=device)
        loss_wavelet = torch.zeros(1, device=device)

        has_pixel_loss = self._pixel_w > 0 or self._lpips_w > 0 or self._wavelet_w > 0
        _temporal_acc = 0.0

        if has_pixel_loss and self._full_decoder is not None:
            from echo_sr.training.perceptual import compute_lpips_loss

            x0_with_grad = model_inputs.video.latent - video_pred
            num_f = latent_meta["num_frames"][0].item()
            h = latent_meta["height"][0].item()
            w = latent_meta["width"][0].item()
            x0_5d = x0_with_grad.view(b, num_f, h, w, 128).permute(0, 4, 1, 2, 3)

            num_decode = self._num_decode_frames
            if num_f <= num_decode:
                frame_indices = list(range(num_f))
            else:
                frame_indices = sorted(random.sample(range(num_f), num_decode))

            _dec_dtype = next(self._full_decoder.parameters()).dtype
            l1_vals = []
            lpips_vals = []

            for idx in frame_indices:
                pred_slice = x0_5d[:, :, idx:idx+1, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_pixels = self._full_decoder(pred_slice.to(_dec_dtype))
                with torch.no_grad():
                    gt_slice = hq_latent[:, :, idx:idx+1, :, :]
                    gt_pixels = self._full_decoder(gt_slice.to(_dec_dtype))

                pred_frame = pred_pixels[:, :, 0]
                gt_frame = gt_pixels[:, :, 0].detach()

                if self._pixel_w > 0:
                    l1_vals.append(F.l1_loss(pred_frame, gt_frame))
                if self._lpips_w > 0 and self._lpips_model is not None:
                    pred_01 = (pred_frame + 1) * 0.5
                    gt_01 = (gt_frame + 1) * 0.5
                    lpips_vals.append(compute_lpips_loss(
                        pred_01.float() * 2 - 1, gt_01.float() * 2 - 1, self._lpips_model,
                    ))

                del pred_pixels, gt_pixels, pred_frame, gt_frame

            if l1_vals:
                loss_pixel = torch.stack(l1_vals).mean()
            if lpips_vals:
                loss_lpips = torch.stack(lpips_vals).mean()

        elif has_pixel_loss and self._tiny_decoder is not None:
            # TinyDecoder fallback path
            from echo_sr.training.perceptual import compute_wavelet_loss, compute_lpips_loss

            x0_with_grad = model_inputs.video.latent - video_pred
            num_f = latent_meta["num_frames"][0].item()
            h = latent_meta["height"][0].item()
            w = latent_meta["width"][0].item()
            x0_5d = x0_with_grad.view(b, num_f, h, w, 128).permute(0, 4, 1, 2, 3)

            pred_ntchw = x0_5d.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
            pred_px = self._tiny_decoder.decode_video(pred_ntchw, parallel=True, show_progress_bar=False)

            gt_px = (hq_video.float() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4)
            out_t = min(pred_px.shape[1], gt_px.shape[1])
            pred_px = pred_px[:, :out_t]
            gt_px = gt_px[:, :out_t].detach()

            if self._pixel_w > 0:
                loss_pixel = F.l1_loss(pred_px, gt_px)

            if self._lpips_w > 0 and self._lpips_model is not None and out_t > 0:
                n_lpips_frames = min(4, out_t)
                if out_t <= n_lpips_frames:
                    frame_indices = list(range(out_t))
                else:
                    step = out_t / n_lpips_frames
                    frame_indices = [int(i * step) for i in range(n_lpips_frames)]
                lpips_losses = []
                for t_idx in frame_indices:
                    l = compute_lpips_loss(pred_px[:, t_idx] * 2 - 1, gt_px[:, t_idx] * 2 - 1, self._lpips_model)
                    lpips_losses.append(l)
                loss_lpips = torch.stack(lpips_losses).mean()

            if self._wavelet_w > 0 and out_t > 0:
                n_wav_frames = min(4, out_t)
                if out_t <= n_wav_frames:
                    wav_indices = list(range(out_t))
                else:
                    wav_step = out_t / n_wav_frames
                    wav_indices = [int(i * wav_step) for i in range(n_wav_frames)]
                pred_wav = pred_px[:, wav_indices].permute(0, 2, 1, 3, 4) * 2 - 1
                gt_wav = gt_px[:, wav_indices].permute(0, 2, 1, 3, 4) * 2 - 1
                loss_wavelet = compute_wavelet_loss(pred_wav, gt_wav)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Combined loss
        # ═══════════════════════════════════════════════════════════════

        # Latent-space temporal consistency loss (zero extra VRAM)
        loss_temporal = torch.zeros(1, device=device)
        if self._temporal_w > 0:
            num_f = latent_meta["num_frames"][0].item()
            if num_f > 1:
                h_lat = latent_meta["height"][0].item()
                w_lat = latent_meta["width"][0].item()
                x0_pred = (model_inputs.video.latent - video_pred).view(b, num_f, h_lat, w_lat, 128)
                x0_gt = hq_latent.permute(0, 2, 3, 4, 1).detach()  # [B, T, H, W, 128]
                pred_diff = x0_pred[:, 1:] - x0_pred[:, :-1]
                gt_diff = x0_gt[:, 1:] - x0_gt[:, :-1]
                loss_temporal = F.l1_loss(pred_diff, gt_diff)
                _temporal_acc = loss_temporal.detach().item()

        # Latent-space frequency loss: high-freq-weighted 3D FFT on latent
        loss_latent_freq = torch.zeros(1, device=device)
        _latent_freq_acc = 0.0
        if self._latent_freq_w > 0:
            num_f = latent_meta["num_frames"][0].item()
            h_lat = latent_meta["height"][0].item()
            w_lat = latent_meta["width"][0].item()
            x0_pred_5d = (model_inputs.video.latent - video_pred).view(b, num_f, h_lat, w_lat, 128).permute(0, 4, 1, 2, 3).float()
            x0_gt_5d = hq_latent.float().detach()  # [B, 128, T, H, W]

            # 3D FFT over (T, H, W)
            pred_fft = torch.fft.rfftn(x0_pred_5d, dim=(2, 3, 4))
            gt_fft = torch.fft.rfftn(x0_gt_5d, dim=(2, 3, 4))

            diff = (pred_fft - gt_fft).abs()
            loss_latent_freq = diff.mean()
            _latent_freq_acc = loss_latent_freq.detach().item()

        total_loss = (
            self._reg_w * loss_reg
            + self._dm_w * loss_dm
            + self._temporal_w * loss_temporal
            + self._latent_freq_w * loss_latent_freq
            + self._pixel_w * loss_pixel
            + self._lpips_w * loss_lpips
            + self._wavelet_w * loss_wavelet
        )

        # Log per-component
        self.strategy._last_pixel_loss = loss_pixel.detach().mean().item() if self._pixel_w > 0 else 0.0
        self.strategy._last_lpips_loss = loss_lpips.detach().mean().item() if self._lpips_w > 0 else 0.0
        self.strategy._last_wavelet_loss = loss_wavelet.detach().mean().item() if self._wavelet_w > 0 else 0.0
        self.strategy._last_temporal_loss = _temporal_acc if self._temporal_w > 0 else 0.0
        self.strategy._last_latent_freq_loss = _latent_freq_acc if self._latent_freq_w > 0 else 0.0
        self.strategy._last_pixel_temporal_loss = 0.0
        self.strategy._last_dm_loss = loss_dm.detach().mean().item() if self._dm_w > 0 else 0.0

        # Store data for critic update (done OUTSIDE accumulate context)
        if self._enable_dmd:
            self._pending_critic_data = (x0_student.detach(), model_inputs)

        if torch.isnan(total_loss.mean()):
            logger.warning(f"Step {self.global_step}: NaN loss → zero")
            total_loss = torch.zeros_like(total_loss)
            total_loss.requires_grad = True

        return total_loss.mean()

    def _update_critic(self) -> None:
        """Critic update: separate forward+backward outside accumulate context."""
        if not self._enable_dmd or not hasattr(self, "_pending_critic_data"):
            return
        if self.global_step % self._critic_interval != 0:
            return

        x0_student, model_inputs = self._pending_critic_data
        device = self.device
        b = x0_student.shape[0]

        self.dmd.activate("critic")

        sigma_c = torch.rand(b, device=device) * 0.9 + 0.05
        noise_c = torch.randn_like(x0_student)
        sigma_c_exp = sigma_c.view(-1, 1, 1)
        noisy_c = (1 - sigma_c_exp) * x0_student + sigma_c_exp * noise_c

        from dataclasses import replace
        critic_mod = replace(
            model_inputs.video,
            latent=noisy_c,
            sigma=sigma_c,
            timesteps=sigma_c.view(-1, 1).expand(-1, noisy_c.shape[1]),
        )
        pred_critic, _ = self.transformer(video=critic_mod, audio=None, perturbations=None)
        target_v = noise_c - x0_student
        loss_critic = F.mse_loss(pred_critic.float(), target_v.float())

        self.critic_optimizer.zero_grad()
        self.accelerator.backward(loss_critic)
        self.accelerator.clip_grad_norm_(self.dmd.critic_parameters(), 1.0)
        self.critic_optimizer.step()

        self.dmd.activate("student")
        self._pending_critic_data = None

    def _log_step(self, loss: torch.Tensor, step_time: float) -> None:
        """Override: log distillation-specific losses."""
        if not self.is_main:
            return

        import json
        cfg = self.config
        opt_cfg = cfg["optimization"]
        log_interval = cfg.get("logging", {}).get("log_interval", 1)
        total_steps = opt_cfg["steps"]

        self._step_times.append(step_time)
        loss_val = loss.item()
        lr_now = self.optimizer.param_groups[0]["lr"]

        v_loss = getattr(self.strategy, "_last_video_loss", 0.0)
        a_loss = getattr(self.strategy, "_last_audio_loss", 0.0)
        dm_loss = getattr(self.strategy, "_last_dm_loss", 0.0)
        px_loss = getattr(self.strategy, "_last_pixel_loss", 0.0)
        lpips_loss = getattr(self.strategy, "_last_lpips_loss", 0.0)
        wav_loss = getattr(self.strategy, "_last_wavelet_loss", 0.0)
        temp_loss = getattr(self.strategy, "_last_temporal_loss", 0.0)
        px_temp_loss = getattr(self.strategy, "_last_pixel_temporal_loss", 0.0)
        lf_loss = getattr(self.strategy, "_last_latent_freq_loss", 0.0)

        readable_loss = (
            self._reg_w * v_loss
            + self._dm_w * dm_loss
            + self._pixel_w * px_loss
            + self._lpips_w * lpips_loss
            + self._wavelet_w * wav_loss
            + self._temporal_w * temp_loss
            + self._pixel_temporal_w * px_temp_loss
            + self._latent_freq_w * lf_loss
        )

        metrics = {
            "train/loss": readable_loss,
            "train/loss_raw": loss_val,
            "train/video_loss": v_loss,
            "train/audio_loss": a_loss,
            "train/dm_loss": dm_loss,
            "train/pixel_loss": px_loss,
            "train/lpips_loss": lpips_loss,
            "train/wavelet_loss": wav_loss,
            "train/temporal_loss": temp_loss,
            "train/pixel_temporal_loss": px_temp_loss,
            "train/latent_freq_loss": lf_loss,
            "train/lr": lr_now,
            "train/grad_norm": self._grad_norm,
            "train/step_time": step_time,
            "train/global_step": self.global_step,
        }

        try:
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            metrics["system/gpu_alloc_gb"] = mem_alloc
            metrics["system/gpu_reserved_gb"] = mem_reserved
        except Exception:
            mem_alloc = mem_reserved = 0.0

        # JSONL
        if self._jsonl_fh is not None:
            record = {
                "step": self.global_step,
                "loss": round(loss_val, 6),
                "video_loss": round(v_loss, 6),
                "audio_loss": round(a_loss, 6),
                "dm_loss": round(dm_loss, 6),
                "pixel_loss": round(px_loss, 6),
                "lpips_loss": round(lpips_loss, 6),
                "wavelet_loss": round(wav_loss, 6),
                "temporal_loss": round(temp_loss, 6),
                "pixel_temporal_loss": round(px_temp_loss, 6),
                "latent_freq_loss": round(lf_loss, 6),
                "lr": lr_now,
                "grad_norm": round(self._grad_norm, 6),
                "step_time": round(step_time, 3),
                "gpu_gb": round(mem_alloc, 2),
                "timestamp": time.time(),
            }
            self._jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.global_step % 10 == 0:
                self._jsonl_fh.flush()

        # SwanLab
        if self._swanlab_run is not None:
            try:
                import swanlab as _swanlab
                _swanlab.log(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=self.global_step,
                )
            except Exception:
                pass

        # Console
        if log_interval > 0 and self.global_step % log_interval == 0:
            avg_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0.0
            remaining = (total_steps - self.global_step) * avg_time
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)

            loss_parts = f"v={v_loss:.5f} a={a_loss:.5f}"
            if dm_loss > 0:
                loss_parts += f" dm={dm_loss:.5f}"
            if px_loss > 0:
                loss_parts += f" px={px_loss:.5f}"
            if lpips_loss > 0:
                loss_parts += f" lpips={lpips_loss:.5f}"
            if wav_loss > 0:
                loss_parts += f" wav={wav_loss:.5f}"
            if temp_loss > 0:
                loss_parts += f" tmp={temp_loss:.5f}"
            if px_temp_loss > 0:
                loss_parts += f" ptmp={px_temp_loss:.5f}"
            if lf_loss > 0:
                loss_parts += f" lfreq={lf_loss:.5f}"

            logger.info(
                f"Step {self.global_step:>6d}/{total_steps} | "
                f"loss={loss_val:.5f} ({loss_parts}) | "
                f"lr={lr_now:.2e} | gn={self._grad_norm:.4f} | "
                f"time={step_time:.2f}s (avg={avg_time:.2f}s) | "
                f"ETA={eta_h}h{eta_m:02d}m | "
                f"GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
            )

        if self.global_step % 100 == 0:
            gc.collect()

    def train(self) -> None:
        """Training loop with critic update outside accumulate context."""
        import time as _time
        cfg = self.config
        opt_cfg = cfg["optimization"]
        total_steps = opt_cfg["steps"]
        max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)
        val_interval = cfg.get("validation", {}).get("interval", 100)
        save_interval = cfg.get("checkpoints", {}).get("interval", 200)

        initial_step = self.global_step
        data_iter = iter(self.dataloader)
        self.transformer.train()

        logger.info(f"Distill training: steps {initial_step}→{total_steps}, "
                    f"val@{val_interval}, save@{save_interval}")

        for step in range(initial_step, total_steps):
            step_start = _time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # DMD distillation: manually manage forward/backward
            # (cannot use accumulate() with multiple transformer forwards)
            loss = self._train_step(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Step {step+1}: NaN/Inf loss — skipping")
                self.optimizer.zero_grad()
                self.global_step = step + 1
                continue

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)

            if max_grad_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    list(self.dmd.student.parameters()) + self.cond_params, max_grad_norm
                )
                if grad_norm is not None:
                    self._grad_norm = float(grad_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step = step + 1

            # Critic update (OUTSIDE accumulate — separate forward+backward)
            self._update_critic()

            step_time = _time.time() - step_start
            self._log_step(loss, step_time)

            if val_interval and self.global_step % val_interval == 0:
                self._validate()

            if save_interval and self.global_step % save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        self._cleanup_logging()
        logger.info("Distillation complete.")

    def _save_checkpoint(self) -> None:
        """Save student LoRA + cond_proj + audio LoRA from teacher."""
        from safetensors.torch import save_file, load_file
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

        self.accelerator.wait_for_everyone()

        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if hasattr(self.transformer, "module"):
            with FSDP.state_dict_type(self.transformer, StateDictType.FULL_STATE_DICT, full_cfg):
                full_sd = self.transformer.state_dict()
        else:
            full_sd = self.transformer.state_dict()

        if not self.is_main:
            return

        save_dir = self.output_dir / "checkpoints"
        save_dir.mkdir(exist_ok=True)
        path = save_dir / f"lora_step_{self.global_step:06d}.safetensors"

        state = {}

        # Extract student LoRA weights from full state dict
        for k, v in full_sd.items():
            if "lora_A" in k or "lora_B" in k:
                clean_k = k.replace(f".{self.lora._adapter_attr}.", ".")
                if not clean_k.endswith(".weight"):
                    clean_k = clean_k + ".weight"
                state[f"diffusion_model.{clean_k}"] = v.detach().cpu().to(torch.bfloat16)
            elif "_cond_" in k and "proj" in k:
                state[f"diffusion_model.{k}"] = v.detach().cpu().to(torch.bfloat16)

        # Bake scaling into B
        for module_name, scaling in self.lora._module_scalings.items():
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            if b_key in state:
                state[b_key] = state[b_key] * scaling

        # Merge untrained LoRA from teacher (audio modules + av_ca gates not in student target_modules)
        teacher_ckpt = self.config["distillation"]["teacher_checkpoint"]
        teacher_sd = load_file(str(teacher_ckpt)) if str(teacher_ckpt).endswith(".safetensors") else torch.load(str(teacher_ckpt), map_location="cpu", weights_only=False)
        for k, v in teacher_sd.items():
            if k not in state and ("lora_A" in k or "lora_B" in k):
                state[k] = v.to(torch.bfloat16)

        save_file(state, str(path))
        n_lora = sum(1 for k in state if "lora_" in k)
        n_cond = sum(1 for k in state if "cond_" in k and "lora_" not in k)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved: {path.name} ({n_lora} LoRA + {n_cond} cond_proj, {size_mb:.1f}MB)")
