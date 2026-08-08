"""EchoTrainer: Clean training loop for AV super-resolution.

No monkey-patching. Direct, readable training loop with:
- On-the-fly VAE encoding
- Fixed prompt text conditioning (no CFG)
- Flow matching (shifted logit-normal σ sampling)
- FSDP distributed training via accelerate
- Rich logging: SwanLab + JSONL + file + console
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from echo_sr import logger
from echo_sr.data.collate import online_collate_fn
from echo_sr.model.lora import EchoLoRA

try:
    import swanlab as _swanlab
except ImportError:
    _swanlab = None


class EchoTrainer:
    """Main trainer for Echo SR."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0

        # Will be initialized in setup()
        self.transformer = None
        self.lora = None
        self.optimizer = None
        self.dataloader = None
        self.accelerator = None
        self.global_step = 0

        # Logging state
        self._swanlab_run = None
        self._jsonl_fh = None
        self._file_handler = None
        self._step_times: deque[float] = deque(maxlen=100)
        self._grad_norm: float = 0.0

    def setup(self) -> None:
        """Initialize all components. Call after __init__."""
        cfg = self.config
        from accelerate import Accelerator

        # Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg["optimization"].get("gradient_accumulation_steps", 1),
            mixed_precision="bf16" if cfg["acceleration"].get("mixed_precision_mode") == "bf16" else "no",
        )
        self.device = self.accelerator.device
        self.is_main = self.accelerator.is_main_process
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes

        # Load model
        from echo_sr.model.loader import (
            load_transformer, load_vae_encoder, load_audio_vae_encoder,
            init_cond_proj, encode_fixed_prompts, load_embeddings_processor,
        )

        model_cfg = cfg["model"]
        self.transformer = load_transformer(model_cfg["model_path"], dtype=torch.bfloat16)

        # Cond proj
        data_cfg = cfg["data"]
        from ltx_core.types import SpatioTemporalScaleFactors
        scale = SpatioTemporalScaleFactors.default()
        hq_w, hq_h, _ = data_cfg["hq_resolution"]
        lq_w, lq_h, _ = data_cfg["lq_resolution"]
        cond_proj_cfg = cfg.get("cond_proj")
        sr_spatial = None
        if cond_proj_cfg and cond_proj_cfg.get("type") == "latent_2x":
            # Resolution-independent 2x mode — no fixed sr_spatial needed
            pass
        elif lq_w != hq_w or lq_h != hq_h:
            sr_spatial = {
                "lq_h": lq_h // scale.height, "lq_w": lq_w // scale.width,
                "hq_h": hq_h // scale.height, "hq_w": hq_w // scale.width,
            }
        self.cond_params = init_cond_proj(self.transformer, sr_spatial, cond_proj_cfg)

        # LoRA
        lora_cfg = cfg["lora"]
        self.lora = EchoLoRA(
            self.transformer,
            target_modules=lora_cfg["target_modules"],
            rank=lora_cfg.get("rank", 384),
            alpha=lora_cfg.get("alpha", 384),
            checkpoint=model_cfg.get("load_checkpoint"),
            lora_structure_from=model_cfg.get("lora_structure_from"),
            param_dtype=torch.float32,
        )

        # Load cond_proj from checkpoint (if available)
        ckpt_path = model_cfg.get("load_checkpoint")
        if ckpt_path and Path(ckpt_path).exists():
            from safetensors.torch import load_file as _load_sf
            ckpt_sd = _load_sf(ckpt_path) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path, map_location="cpu", weights_only=False)
            loaded_cond = 0
            for name, param in self.transformer.named_parameters():
                if "_cond_" in name and "proj" in name:
                    key = f"diffusion_model.{name}"
                    if key in ckpt_sd:
                        param.data.copy_(ckpt_sd[key].to(dtype=param.dtype))
                        loaded_cond += 1
            if loaded_cond > 0:
                logger.info(f"Loaded cond_proj weights: {loaded_cond} tensors from {Path(ckpt_path).name}")

        # Cast transformer to float32 for FSDP uniform dtype
        self.transformer = self.transformer.to(dtype=torch.float32)
        self.transformer.set_gradient_checkpointing(cfg["optimization"].get("enable_gradient_checkpointing", True))

        # Count params BEFORE FSDP (full, not sharded)
        self._total_lora_params = self.lora.num_parameters()
        self._total_cond_params = sum(p.numel() for p in self.cond_params)
        self._total_trainable = self._total_lora_params + self._total_cond_params

        # torch.compile disabled — incompatible with FSDP full-shard + gradient checkpointing
        # (dynamo traces fake tensors with shape 0 for sharded parameters like scale_shift_table)
        # from ltx_core.model.transformer.compiling import compile_transformer
        # self.transformer = compile_transformer(self.transformer)
        # logger.info("Transformer blocks compiled with torch.compile")

        # FSDP prepare
        self.transformer = self.accelerator.prepare(self.transformer)

        # Optimizer (LoRA + cond_proj) — uses FSDP-sharded param references
        trainable_params = list(self.lora.parameters()) + self.cond_params
        self._trainable_params = trainable_params
        opt_cfg = cfg["optimization"]
        opt_params = cfg.get("optimizer_params", {})
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(opt_cfg["learning_rate"]),
            betas=tuple(opt_params.get("betas", [0.9, 0.999])),
            weight_decay=float(opt_params.get("weight_decay", 0.01)),
        )

        # LR Scheduler
        self.scheduler = None
        scheduler_type = opt_cfg.get("scheduler_type")
        if scheduler_type:
            from transformers import get_scheduler
            scheduler_params = opt_cfg.get("scheduler_params", {})
            self.scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=scheduler_params.get("warmup_steps", 0),
                num_training_steps=opt_cfg["steps"],
            )
            self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # VAE encoders
        self.vae_encoder = load_vae_encoder(model_cfg["model_path"], self.device)
        self.audio_vae_encoder = None
        if cfg["training_strategy"].get("with_audio", True):
            self.audio_vae_encoder = load_audio_vae_encoder(model_cfg["model_path"], self.device)

        # Text embeddings — precompute Block 1+2+3 once (fixed prompt)
        from echo_sr.training.prompts import SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT
        cache_path = Path(cfg.get("prompt_cache_path", "checkpoints/echo-sr/prompt/sr_prompt_embeddings.pt"))
        self.cond_feats, self.val_embeds = encode_fixed_prompts(
            model_cfg["model_path"], model_cfg["text_encoder_path"], self.device,
            SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT, cache_path,
        )
        logger.info("Prompt embeddings ready (fixed SR prompt, no CFG)")

        # Strategy
        from echo_sr.training.strategy import AVRestorationConfig, AVRestorationStrategy
        strategy_cfg = cfg["training_strategy"]
        self.strategy = AVRestorationStrategy(AVRestorationConfig(**strategy_cfg))

        # Timestep sampler
        from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler
        self.timestep_sampler = ShiftedLogitNormalTimestepSampler()

        # Dataset
        from echo_sr.data.online_dataset import OnlineAVDataset
        max_f = data_cfg["hq_resolution"][2]
        self.dataset = OnlineAVDataset(
            data_root=data_cfg["preprocessed_data_root"],
            hq_width=hq_w, hq_height=hq_h,
            lq_width=lq_w, lq_height=lq_h,
            target_frames=max_f,
            with_audio=strategy_cfg.get("with_audio", True),
            seed=cfg.get("seed", 42),
        )

        # Dataloader
        sampler = DistributedSampler(
            self.dataset, num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index, shuffle=True, drop_last=True,
        ) if self.accelerator.num_processes > 1 else None

        num_workers = data_cfg.get("num_dataloader_workers", 0)
        self.dataloader = DataLoader(
            self.dataset, batch_size=opt_cfg.get("batch_size", 1),
            shuffle=(sampler is None), sampler=sampler, drop_last=True,
            num_workers=num_workers,
            collate_fn=online_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Output dir (broadcast timestamp from rank 0 for multi-node consistency)
        output_dir = cfg.get("output_dir", "outputs/echo_sr")
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

        # Setup logging (file + SwanLab + JSONL)
        self._setup_logging()

        # Print start banner
        self._log_start_banner()

    # ─────────────────────────────────────────────────────────────────────────
    # Logging Setup
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """Initialize file logger, SwanLab, JSONL, and config snapshot."""
        cfg = self.config
        logging_cfg = cfg.get("logging", {})
        swanlab_cfg = cfg.get("swanlab", {})

        if not self.is_main:
            return

        # File logger
        log_path = self.output_dir / "training.log"
        self._file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(self._file_handler)
        logger.info(f"File logging: {log_path}")

        # Per-step JSONL log
        jsonl_path = self.output_dir / "step_log.jsonl"
        self._jsonl_fh = open(jsonl_path, "a", encoding="utf-8")
        logger.info(f"JSONL step log: {jsonl_path}")

        # Config snapshot (single file, overwrite on restart)
        config_snapshot = self.output_dir / "config.yaml"
        with open(config_snapshot, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved: {config_snapshot}")

        # SwanLab
        if swanlab_cfg.get("enabled", False):
            if _swanlab is None:
                logger.warning("swanlab not installed — pip install swanlab")
            else:
                api_key = swanlab_cfg.get("api_key")
                if api_key and not os.environ.get("SWANLAB_API_KEY"):
                    os.environ["SWANLAB_API_KEY"] = str(api_key)

                exp_name = swanlab_cfg.get("experiment_name")
                if not exp_name:
                    exp_name = f"{self.output_dir.name}_{datetime.now().strftime('%m%d_%H%M')}"

                mode = swanlab_cfg.get("mode", "online" if api_key else "offline")
                swanlab_logdir = str(self.output_dir / "swanlab")
                os.makedirs(swanlab_logdir, exist_ok=True)

                try:
                    init_kwargs = {
                        "project": swanlab_cfg.get("project", "Echo_SR_Training"),
                        "experiment_name": exp_name,
                        "config": cfg,
                        "logdir": swanlab_logdir,
                        "mode": mode,
                    }
                    if api_key:
                        init_kwargs["api_key"] = api_key
                    tags = swanlab_cfg.get("tags")
                    if tags:
                        init_kwargs["tags"] = tags

                    self._swanlab_run = _swanlab.init(**init_kwargs)
                    logger.info(f"SwanLab: project={init_kwargs['project']}, exp={exp_name}, mode={mode}")
                except Exception as exc:
                    logger.warning(f"SwanLab init failed: {exc}")

        # Disable Python GC for stable step times
        gc.disable()
        logger.info("Python GC disabled (manual gc.collect() every 100 steps)")

    def _log_start_banner(self) -> None:
        """Print a comprehensive start banner."""
        if not self.is_main:
            return

        cfg = self.config
        opt_cfg = cfg["optimization"]
        data_cfg = cfg["data"]
        lora_cfg = cfg["lora"]
        strategy_cfg = cfg["training_strategy"]
        total_params = self._total_trainable
        lora_params = self._total_lora_params
        cond_params = self._total_cond_params

        logger.info("=" * 72)
        logger.info("Echo SR Training — AV Super-Resolution (SFT)")
        logger.info("=" * 72)
        logger.info(f"  Output dir:        {self.output_dir}")
        logger.info(f"  Model:             {cfg['model']['model_path']}")
        logger.info(f"  LoRA structure:    {cfg['model'].get('lora_structure_from', 'from config')}")
        logger.info(f"  Resume ckpt:       {cfg['model'].get('load_checkpoint') or 'none (fresh start)'}")
        logger.info(f"  Resolution:        LQ {data_cfg['lq_resolution']} → HQ {data_cfg['hq_resolution']}")
        logger.info(f"  Dataset:           {len(self.dataset)} samples from {data_cfg['preprocessed_data_root']}")
        logger.info(f"  ──── Training ────")
        logger.info(f"  Total steps:       {opt_cfg['steps']}")
        logger.info(f"  Batch size:        {opt_cfg.get('batch_size', 1)} x {opt_cfg.get('gradient_accumulation_steps', 1)} accum x {self.world_size} GPUs")
        logger.info(f"  Learning rate:     {opt_cfg['learning_rate']}")
        logger.info(f"  LR scheduler:      {opt_cfg.get('scheduler_type', 'none')}")
        logger.info(f"  Max grad norm:     {opt_cfg.get('max_grad_norm', 1.0)}")
        logger.info(f"  Mixed precision:   {cfg['acceleration'].get('mixed_precision_mode', 'no')}")
        logger.info(f"  Grad checkpoint:   {opt_cfg.get('enable_gradient_checkpointing', True)}")
        logger.info(f"  ──── Model ────")
        logger.info(f"  LoRA rank:         {lora_cfg.get('rank', 384)}, alpha={lora_cfg.get('alpha', 384)}")
        logger.info(f"  LoRA modules:      {len(self.lora._targets)}")
        logger.info(f"  LoRA params:       {lora_params:,}")
        logger.info(f"  Cond proj params:  {cond_params:,}")
        logger.info(f"  Total trainable:   {total_params:,}")
        logger.info(f"  ──── Strategy ────")
        logger.info(f"  LQ dropout:        {strategy_cfg.get('lq_drop_prob', 0.1)}")
        logger.info(f"  Cond noise:        [{strategy_cfg.get('condition_noise_min', 0)}, {strategy_cfg.get('condition_noise_max', 0.6)}]")
        logger.info(f"  Audio cond noise:  [{strategy_cfg.get('audio_condition_noise_min', 0)}, {strategy_cfg.get('audio_condition_noise_max', 0.6)}]")
        logger.info(f"  Audio loss weight: {strategy_cfg.get('audio_loss_weight', 2.0)}")
        logger.info(f"  First-frame cond:  {strategy_cfg.get('first_frame_conditioning_p', 0.5)}")
        logger.info(f"  ──── Logging ────")
        logger.info(f"  SwanLab:           {'enabled' if self._swanlab_run else 'disabled'}")
        logger.info(f"  Log interval:      every {cfg.get('logging', {}).get('log_interval', 1)} step(s)")
        logger.info(f"  Val interval:      every {cfg.get('validation', {}).get('interval', 100)} step(s)")
        logger.info(f"  Save interval:     every {cfg.get('checkpoints', {}).get('interval', 200)} step(s)")
        logger.info("=" * 72)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-step Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_step(self, loss: torch.Tensor, step_time: float) -> None:
        """Rich per-step logging: console + JSONL + SwanLab."""
        if not self.is_main:
            return

        cfg = self.config
        opt_cfg = cfg["optimization"]
        log_interval = cfg.get("logging", {}).get("log_interval", 1)
        total_steps = opt_cfg["steps"]

        self._step_times.append(step_time)
        loss_val = loss.item()
        lr_now = self.optimizer.param_groups[0]["lr"]

        # Per-modality loss from strategy
        v_loss = getattr(self.strategy, "_last_video_loss", 0.0)
        a_loss = getattr(self.strategy, "_last_audio_loss", 0.0)

        # Metrics dict
        metrics = {
            "train/loss": loss_val,
            "train/video_loss": v_loss,
            "train/audio_loss": a_loss,
            "train/lr": lr_now,
            "train/grad_norm": self._grad_norm,
            "train/step_time": step_time,
            "train/global_step": self.global_step,
        }

        # GPU memory
        try:
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            metrics["system/gpu_alloc_gb"] = mem_alloc
            metrics["system/gpu_reserved_gb"] = mem_reserved
        except Exception:
            mem_alloc = mem_reserved = 0.0

        # JSONL log (every step)
        if self._jsonl_fh is not None:
            record = {
                "step": self.global_step,
                "loss": round(loss_val, 6),
                "video_loss": round(v_loss, 6),
                "audio_loss": round(a_loss, 6),
                "lr": lr_now,
                "grad_norm": round(self._grad_norm, 6),
                "step_time": round(step_time, 3),
                "gpu_gb": round(mem_alloc, 2),
                "timestamp": time.time(),
            }
            self._jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.global_step % 10 == 0:
                self._jsonl_fh.flush()

        # SwanLab (every step)
        if self._swanlab_run is not None:
            try:
                _swanlab.log(
                    {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=self.global_step,
                )
            except Exception:
                pass

        # Console + file log (every log_interval)
        if log_interval > 0 and self.global_step % log_interval == 0:
            avg_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0.0
            remaining = (total_steps - self.global_step) * avg_time
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)

            logger.info(
                f"Step {self.global_step:>6d}/{total_steps} | "
                f"loss={loss_val:.5f} (v={v_loss:.5f} a={a_loss:.5f}) | "
                f"lr={lr_now:.2e} | gn={self._grad_norm:.4f} | "
                f"time={step_time:.2f}s (avg={avg_time:.2f}s) | "
                f"ETA={eta_h}h{eta_m:02d}m | "
                f"GPU: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
            )

        # Periodic GC
        if self.global_step % 100 == 0:
            gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Main training loop."""
        cfg = self.config
        opt_cfg = cfg["optimization"]
        total_steps = opt_cfg["steps"]
        max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)
        val_interval = cfg.get("validation", {}).get("interval", 100)
        save_interval = cfg.get("checkpoints", {}).get("interval", 200)

        initial_step = self.global_step
        data_iter = iter(self.dataloader)
        self.transformer.train()

        logger.info(f"Training: steps {initial_step}→{total_steps}, "
                    f"val@{val_interval}, save@{save_interval}")

        for step in range(initial_step, total_steps):
            step_start = time.time()

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Training step
            with self.accelerator.accumulate(self.transformer):
                loss = self._train_step(batch)

                # NaN protection: skip step if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Step {step+1}: NaN/Inf loss detected — skipping optimizer step")
                    self.optimizer.zero_grad()
                    self.global_step = step + 1
                    step_time = time.time() - step_start
                    self._log_step(loss, step_time)
                    continue

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and max_grad_norm > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        list(self.lora.parameters()) + self.cond_params, max_grad_norm
                    )
                    if grad_norm is not None:
                        self._grad_norm = float(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            self.global_step = step + 1
            step_time = time.time() - step_start

            # Logging
            self._log_step(loss, step_time)

            # Validation
            if val_interval and self.global_step % val_interval == 0:
                self._validate()

            # Checkpoint
            if save_interval and self.global_step % save_interval == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint()
        self._cleanup_logging()
        logger.info("Training complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Training Step
    # ─────────────────────────────────────────────────────────────────────────

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single training step: VAE encode → strategy forward → loss."""
        device = self.device
        is_pixel_wave = self.config.get("cond_proj", {}).get("type") == "pixel_wave"

        from ltx_core.model.video_vae import TilingConfig
        enc_tiling = TilingConfig.default()

        # Video VAE encoding
        hq_video = batch["hq_video"].to(device=device, dtype=torch.bfloat16)
        lq_video = batch["lq_video"].to(device=device, dtype=torch.bfloat16)
        if hq_video.dim() == 4:
            hq_video = hq_video.unsqueeze(0)
            lq_video = lq_video.unsqueeze(0)

        # Degradation (keep in bf16 to avoid fp32 memory spike)
        from echo_sr.data.video_degradation import apply_video_degradation
        with torch.no_grad():
            rng = random.Random(42 + self.global_step * self.world_size + self.rank)
            lq_video = apply_video_degradation(lq_video.float(), rng=rng, downsample_factor=4).to(torch.bfloat16)

        # VAE encode HQ (always needed as denoising target)
        with torch.no_grad():
            hq_latent = self.vae_encoder.tiled_encode(hq_video, tiling_config=enc_tiling)
            if not is_pixel_wave:
                lq_latent = self.vae_encoder.tiled_encode(lq_video, tiling_config=enc_tiling)

        b = hq_latent.shape[0]
        latent_meta = {
            "num_frames": torch.tensor([hq_latent.shape[2]] * b),
            "height": torch.tensor([hq_latent.shape[3]] * b),
            "width": torch.tensor([hq_latent.shape[4]] * b),
            "fps": batch["fps"] if isinstance(batch["fps"], torch.Tensor) else torch.tensor([batch["fps"]] * b),
        }

        # Audio encoding
        hq_audio_lat = lq_audio_lat = None
        if self.audio_vae_encoder is not None:
            hq_audio = batch.get("hq_audio")
            lq_audio = batch.get("lq_audio")
            if hq_audio is not None and not (hq_audio == 0).all():
                from ltx_core.model.audio_vae import encode_audio
                from ltx_core.types import Audio
                audio_sr = batch["audio_sr"]
                if isinstance(audio_sr, torch.Tensor):
                    audio_sr = int(audio_sr[0].item())
                with torch.no_grad():
                    hq_audio_lat = self._encode_audio(hq_audio, audio_sr)
                    if not is_pixel_wave:
                        lq_audio_lat = self._encode_audio(lq_audio, audio_sr)

        # Text conditions — fixed SR prompt
        feats = self.cond_feats
        conditions = {
            "video_prompt_embeds": feats["video_embeds"][0:1].expand(b, -1, -1).to(device),
            "prompt_attention_mask": feats["attention_mask"][0:1].expand(b, -1).to(device),
        }
        if feats["audio_embeds"] is not None:
            conditions["audio_prompt_embeds"] = feats["audio_embeds"][0:1].expand(b, -1, -1).to(device)

        # Assemble batch
        assembled = {
            "hq_latents": {"latents": hq_latent, **latent_meta},
            "conditions": conditions,
        }

        if is_pixel_wave:
            # pixel_wave: pass raw pixel and waveform directly (no LQ VAE encode)
            assembled["lq_pixel"] = lq_video  # [B, 3, T, H_lq, W_lq] bf16
            if self.audio_vae_encoder is not None:
                lq_audio = batch.get("lq_audio")
                if lq_audio is not None and not (lq_audio == 0).all():
                    # Ensure stereo [B, 2, T_samples]
                    if lq_audio.dim() == 2:
                        lq_audio = lq_audio.unsqueeze(0)
                    if lq_audio.shape[1] == 1:
                        lq_audio = lq_audio.expand(-1, 2, -1)
                    assembled["lq_waveform"] = lq_audio.to(device=device, dtype=torch.bfloat16)
                else:
                    # Dummy waveform for missing audio
                    assembled["lq_waveform"] = torch.zeros(b, 2, 213444, device=device, dtype=torch.bfloat16)
        else:
            assembled["lq_latents"] = {"latents": lq_latent, **latent_meta}

        if self.audio_vae_encoder is not None:
            if hq_audio_lat is not None:
                assembled["audio_hq_latents"] = {"latents": hq_audio_lat}
                if not is_pixel_wave:
                    assembled["audio_lq_latents"] = {"latents": lq_audio_lat}
            else:
                dummy = torch.zeros(b, 8, 1, 16, device=device, dtype=torch.bfloat16)
                assembled["audio_hq_latents"] = {"latents": dummy}
                if not is_pixel_wave:
                    assembled["audio_lq_latents"] = {"latents": dummy}

        del hq_video
        if not is_pixel_wave:
            del lq_video

        # Store for TRAIN validation (include raw audio for lossless playback)
        self._last_batch = assembled
        self._last_batch_raw_audio = {
            "hq_audio": batch.get("hq_audio"),
            "lq_audio": batch.get("lq_audio"),
            "audio_sr": batch.get("audio_sr"),
        }

        # Strategy forward + loss
        model_inputs = self.strategy.prepare_training_inputs(assembled, self.timestep_sampler)
        video_pred, audio_pred = self.transformer(
            video=model_inputs.video, audio=model_inputs.audio, perturbations=None,
        )
        loss = self.strategy.compute_loss(video_pred, audio_pred, model_inputs)
        return loss.mean()

    def _encode_audio(self, waveforms: torch.Tensor, sr: int) -> torch.Tensor:
        """Encode batch of audio waveforms."""
        from ltx_core.model.audio_vae import encode_audio
        from ltx_core.types import Audio
        results = []
        for i in range(waveforms.shape[0]):
            w = waveforms[i]
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if w.shape[0] == 1:
                w = w.expand(2, -1)
            w = w.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
            lat = encode_audio(Audio(waveform=w, sampling_rate=sr), self.audio_vae_encoder)
            results.append(lat.squeeze(0))
        return torch.stack(results)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation & Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Run validation: all ranks denoise in parallel, each saves its own clip."""
        if not hasattr(self, "_validator"):
            from echo_sr.validation.validator import EchoValidator
            self._validator = EchoValidator(self)
        self.accelerator.wait_for_everyone()
        self.transformer.eval()
        self._validator.run()
        self.transformer.train()
        self.accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

    def _save_checkpoint(self) -> None:
        """Save LoRA + cond_proj in one safetensors file."""
        from safetensors.torch import save_file
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

        self.accelerator.wait_for_everyone()

        # Get full state dict (all ranks contribute, only rank 0 gets the result)
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

        # Extract LoRA weights from full state dict
        for k, v in full_sd.items():
            if "lora_A" in k or "lora_B" in k:
                # Remove adapter attr prefix: xxx._echo_lora_default.lora_A → xxx.lora_A.weight
                clean_k = k.replace(f".{self.lora._adapter_attr}.", ".")
                # Ensure .weight suffix for official format compatibility
                if not clean_k.endswith(".weight"):
                    clean_k = clean_k + ".weight"
                state[f"diffusion_model.{clean_k}"] = v.detach().cpu().to(torch.bfloat16)
            elif "_cond_" in k and "proj" in k:
                state[f"diffusion_model.{k}"] = v.detach().cpu().to(torch.bfloat16)

        # Bake scaling into B for official format compatibility
        for module_name, scaling in self.lora._module_scalings.items():
            b_key = f"diffusion_model.{module_name}.lora_B.weight"
            if b_key in state:
                state[b_key] = state[b_key] * scaling

        save_file(state, str(path))
        n_lora = sum(1 for k in state if "lora_" in k)
        n_cond = sum(1 for k in state if "cond_" in k and "lora_" not in k)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved: {path.name} ({n_lora} LoRA + {n_cond} cond_proj, {size_mb:.1f}MB)")

        # Log to SwanLab
        if self._swanlab_run is not None:
            try:
                _swanlab.log({"checkpoint/step": self.global_step, "checkpoint/size_mb": size_mb}, step=self.global_step)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def _cleanup_logging(self) -> None:
        """Close file handles and finish SwanLab."""
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None
        if self._file_handler is not None:
            logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        if self._swanlab_run is not None:
            try:
                _swanlab.finish()
            except Exception:
                pass
            self._swanlab_run = None
