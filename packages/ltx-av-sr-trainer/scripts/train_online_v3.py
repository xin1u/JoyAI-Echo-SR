#!/usr/bin/env python3
"""On-the-fly training script for AV restoration — v3 fixed prompt.

Same as train_online.py (v2) but uses a fixed SR-quality prompt instead of
per-video captions. This eliminates the need for a text encoder during
inference, making the model suitable for long-video streaming SR pipelines.

Training text conditioning:
  - 90%: fixed SR_FIXED_PROMPT (quality-oriented, content-agnostic)
  - 10%: empty string (CFG dropout)
Validation CFG:
  - positive: SR_FIXED_PROMPT (same as training)
  - negative: DEFAULT_NEGATIVE_PROMPT

After encoding the fixed prompts at startup, the Gemma text encoder and
feature extractor are fully unloaded — saving ~14GB GPU VRAM vs v2.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_av_sr_1k.sh

The launcher sets the PYTHONPATH for the 1.1 vendored snapshot and passes
configs/av_sr_1k_multistep.yaml. Modified for the portable JoyAI-Echo-SR
release in 2026.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import (  # noqa: E402
    SRTrainer,
    _ensure_vae_encoder,
    patch_cross_attention_grad_isolation,
    patch_logging,
    patch_strategy_registry,
    patch_validation,
)

from ltx_av_sr_trainer import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed prompts
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

SR_FIXED_PROMPT = (
    "Extremely sharp and in focus, perfectly exposed with ideal brightness, high contrast, "
    "vibrant saturated colors, minimal noise, clean smooth texture, excellent natural lighting, "
    "stable without flickering, no motion blur, correct anatomical proportions, realistic natural "
    "skin tones, symmetrical well-defined facial features, correct number of limbs, properly formed "
    "hands with five fingers, clean readable text, consistent accurate perspective, perfectly stable "
    "camera, correct depth of field with natural bokeh, clean uncluttered background, no distracting "
    "reflections, soft natural shadows, consistent lighting direction, smooth color gradients without "
    "banding, photorealistic rendering, physically accurate materials, natural human appearance, "
    "correct ethnicity and gender representation, subtle natural expressions, accurate gaze direction, "
    "perfectly synchronized lip movements, clear natural voice audio, clean undistorted sound, no echo "
    "or background noise, perfectly synchronized audio and video, natural authentic dialogue, smooth "
    "fluid movement, natural pacing and timing, seamless transitions, consistent professional framing, "
    "level horizon, dimensional cinematic lighting, consistent visual tone, natural color grading, "
    "unfiltered organic look, and no digital artifacts."
)


# ---------------------------------------------------------------------------
# Fixed embedding setup — encode once, unload Gemma
# ---------------------------------------------------------------------------

PROMPT_CACHE_PATH = Path(
    os.environ.get("AV_SR_PROMPT_CACHE", "checkpoints/prompt/sr_prompt_embeddings.pt")
)


def setup_fixed_embeddings(
    model_path: str,
    text_encoder_path: str,
    device: torch.device,
    sr_prompt: str = SR_FIXED_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
):
    """Load cached prompt embeddings if available, otherwise encode and save.

    Cache path: $AV_SR_PROMPT_CACHE (default: checkpoints/prompt/sr_prompt_embeddings.pt)
    If the cache exists AND the prompt text matches, skip text encoder entirely.
    Otherwise, encode, save cache, then unload text encoder.

    Returns:
        embeddings_processor: Connectors only (feature_extractor=None), on GPU
        cached_cond_feats: dict with video_feats, audio_feats, mask for SR prompt
        cached_empty_feats: dict with video_feats, audio_feats, mask for empty string
        cached_val_embeds: CachedPromptEmbeddings for validation CFG
    """
    from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
    from ltx_trainer.validation_sampler import CachedPromptEmbeddings

    # --- Try loading from cache ---
    if PROMPT_CACHE_PATH.exists():
        logger.info(f"Found cached prompt embeddings: {PROMPT_CACHE_PATH}")
        cached = torch.load(PROMPT_CACHE_PATH, map_location="cpu", weights_only=False)

        # Verify prompt text matches (skip re-encoding if unchanged)
        if cached.get("sr_prompt_text") == sr_prompt and cached.get("negative_prompt_text") == negative_prompt:
            logger.info("Prompt text matches cache — skipping text encoder loading entirely")

            cached_cond_feats = cached["sr_prompt"]
            cached_empty_feats = cached["empty_prompt"]
            cached_val_embeds = CachedPromptEmbeddings(
                video_context_positive=cached["val_positive"]["video_encoding"],
                audio_context_positive=cached["val_positive"]["audio_encoding"],
                video_context_negative=cached["val_negative"]["video_encoding"],
                audio_context_negative=cached["val_negative"]["audio_encoding"],
            )

            # Still need embeddings_processor (connectors) for _training_step
            embeddings_processor = load_embeddings_processor(model_path, device=device, dtype=torch.bfloat16)
            embeddings_processor.requires_grad_(False).eval()
            embeddings_processor.feature_extractor = None

            logger.info(
                f"Loaded from cache. SR cond: v={list(cached_cond_feats['video_feats'].shape)}, "
                f"No text encoder needed (saved ~14GB GPU + ~30s load time)"
            )
            return embeddings_processor, cached_cond_feats, cached_empty_feats, cached_val_embeds
        else:
            logger.info("Prompt text changed — re-encoding with text encoder")

    # --- Encode from scratch ---
    logger.info(f"Loading text encoder from {text_encoder_path}")

    text_encoder = load_text_encoder(
        gemma_model_path=text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=False,
    )
    text_encoder.requires_grad_(False).eval()

    embeddings_processor = load_embeddings_processor(model_path, device=device, dtype=torch.bfloat16)
    embeddings_processor.requires_grad_(False).eval()

    with torch.no_grad():
        # Training features (Block 1+2 only, Block 3 applied in _training_step)
        sr_hs, sr_mask = text_encoder.encode(sr_prompt, padding_side="left")
        sr_v_feats, sr_a_feats = embeddings_processor.feature_extractor(sr_hs, sr_mask, "left")

        cached_cond_feats = {
            "video_feats": sr_v_feats.cpu(),
            "audio_feats": sr_a_feats.cpu() if sr_a_feats is not None else None,
            "mask": sr_mask.cpu(),
        }

        # Empty string → uncond features
        empty_hs, empty_mask = text_encoder.encode("", padding_side="left")
        empty_v_feats, empty_a_feats = embeddings_processor.feature_extractor(empty_hs, empty_mask, "left")

        cached_empty_feats = {
            "video_feats": empty_v_feats.cpu(),
            "audio_feats": empty_a_feats.cpu() if empty_a_feats is not None else None,
            "mask": empty_mask.cpu(),
        }

        # Validation embeddings (Block 1+2+3, full pipeline)
        pos_out = embeddings_processor.process_hidden_states(sr_hs, sr_mask, padding_side="left")
        neg_out = embeddings_processor.process_hidden_states(
            *text_encoder.encode(negative_prompt, padding_side="left"), padding_side="left"
        )

    cached_val_embeds = CachedPromptEmbeddings(
        video_context_positive=pos_out.video_encoding.cpu(),
        audio_context_positive=pos_out.audio_encoding.cpu(),
        video_context_negative=neg_out.video_encoding.cpu(),
        audio_context_negative=(
            neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None
        ),
    )

    logger.info(
        f"Fixed prompts encoded. "
        f"SR cond: v={list(sr_v_feats.shape)}, "
        f"Empty: v={list(empty_v_feats.shape)}, "
        f"Val neg: v={list(neg_out.video_encoding.shape)}"
    )

    # --- Unload text encoder + feature extractor ---
    del text_encoder
    embeddings_processor.feature_extractor = None
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Text encoder + feature extractor unloaded (saved ~14GB GPU)")

    # --- Save to cache for future runs ---
    PROMPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sr_prompt": cached_cond_feats,
        "empty_prompt": cached_empty_feats,
        "val_positive": {
            "video_encoding": cached_val_embeds.video_context_positive,
            "audio_encoding": cached_val_embeds.audio_context_positive,
        },
        "val_negative": {
            "video_encoding": cached_val_embeds.video_context_negative,
            "audio_encoding": cached_val_embeds.audio_context_negative,
        },
        "sr_prompt_text": sr_prompt,
        "negative_prompt_text": negative_prompt,
    }, PROMPT_CACHE_PATH)
    logger.info(f"Prompt embeddings cached to: {PROMPT_CACHE_PATH}")

    return embeddings_processor, cached_cond_feats, cached_empty_feats, cached_val_embeds


# ---------------------------------------------------------------------------
# On-the-fly training step wrapper (fixed prompt version)
# ---------------------------------------------------------------------------

def _make_online_training_step_v3(
    trainer,
    vae_encoder,
    audio_vae_encoder,
    cached_cond_feats: dict,
    cached_empty_feats: dict,
    with_audio: bool,
    cfg_rate: float = 0.1,
    rank: int = 0,
    world_size: int = 1,
):
    """Monkey-patch _training_step with fixed prompt embeddings.

    Same as v2 but replaces per-video text encoding with pre-cached tensors.
    """
    from ltx_core.model.audio_vae import encode_audio
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.types import Audio
    from ltx_av_sr_trainer.degradation.video_degradation import apply_video_degradation

    enc_tiling = TilingConfig.default()
    original_training_step = trainer._training_step.__func__
    _degrade_step = [0]

    def _online_training_step(self, batch):
        device = self._accelerator.device
        from ltx_av_sr_trainer import logger as _logger

        # --- Video VAE encoding ---
        hq_video = batch["hq_video"].to(device=device, dtype=torch.bfloat16)
        lq_video = batch["lq_video"].to(device=device, dtype=torch.bfloat16)

        if hq_video.dim() == 4:
            hq_video = hq_video.unsqueeze(0)
            lq_video = lq_video.unsqueeze(0)

        # Rank-dependent degradation seed
        with torch.no_grad():
            rng = random.Random(42 + _degrade_step[0] * world_size + rank)
            _degrade_step[0] += 1
            lq_video = apply_video_degradation(lq_video.float(), rng=rng, downsample_factor=1).to(torch.bfloat16)

        with torch.no_grad():
            hq_latent = vae_encoder.tiled_encode(hq_video, tiling_config=enc_tiling)
            lq_latent = vae_encoder.tiled_encode(lq_video, tiling_config=enc_tiling)
        del hq_video, lq_video
        torch.cuda.empty_cache()

        b = hq_latent.shape[0]
        latent_meta = {
            "num_frames": torch.tensor([hq_latent.shape[2]] * b),
            "height": torch.tensor([hq_latent.shape[3]] * b),
            "width": torch.tensor([hq_latent.shape[4]] * b),
            "fps": batch["fps"] if isinstance(batch["fps"], torch.Tensor) else torch.tensor([batch["fps"]] * b),
        }

        # --- Fixed prompt selection (no text encoder needed) ---
        cfg_rng = random.Random(42 + _degrade_step[0] * world_size + rank + 7777)
        if cfg_rng.random() < cfg_rate:
            feats = cached_empty_feats
            _logger.info("CFG dropout: using empty prompt embedding")
        else:
            feats = cached_cond_feats
            _logger.info("Using fixed SR prompt embedding")

        conditions = {
            "video_prompt_embeds": feats["video_feats"][0:1].expand(b, -1, -1).to(device),
            "prompt_attention_mask": feats["mask"][0:1].expand(b, -1).to(device),
        }
        if feats["audio_feats"] is not None:
            conditions["audio_prompt_embeds"] = feats["audio_feats"][0:1].expand(b, -1, -1).to(device)

        # --- Audio VAE encoding ---
        hq_audio_lat = lq_audio_lat = None
        raw_hq_audio = raw_lq_audio = None
        raw_audio_sr = None

        if with_audio and audio_vae_encoder is not None:
            hq_audio = batch.get("hq_audio")
            lq_audio = batch.get("lq_audio")
            audio_sr = batch["audio_sr"]
            if isinstance(audio_sr, torch.Tensor):
                audio_sr = int(audio_sr[0].item())

            if hq_audio is not None and not (hq_audio == 0).all():
                _logger.info(
                    f"Audio: hq shape={list(hq_audio.shape)} rms={hq_audio.float().pow(2).mean().sqrt():.4f} "
                    f"max={hq_audio.abs().max():.4f} sr={audio_sr}"
                )
                with torch.no_grad():
                    def _encode_audio_batch(waveforms, sr):
                        results = []
                        for i in range(waveforms.shape[0]):
                            w = waveforms[i]
                            if w.dim() == 1:
                                w = w.unsqueeze(0)
                            if w.shape[0] == 1:
                                w = w.expand(2, -1)
                            w = w.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                            audio_obj = Audio(waveform=w, sampling_rate=sr)
                            lat = encode_audio(audio_obj, audio_vae_encoder)
                            results.append(lat.squeeze(0))
                        return torch.stack(results)

                    hq_audio_lat = _encode_audio_batch(hq_audio, audio_sr)
                    lq_audio_lat = _encode_audio_batch(lq_audio, audio_sr)
                    raw_hq_audio = hq_audio[0].to(device)
                    raw_lq_audio = lq_audio[0].to(device)
                    raw_audio_sr = audio_sr

        # --- Assemble batch ---
        assembled = {
            "hq_latents": {"latents": hq_latent, **latent_meta},
            "lq_latents": {"latents": lq_latent, **latent_meta},
            "conditions": conditions,
        }

        if with_audio:
            if hq_audio_lat is not None:
                assembled["audio_hq_latents"] = {"latents": hq_audio_lat}
                assembled["audio_lq_latents"] = {"latents": lq_audio_lat}
                assembled["raw_hq_audio"] = raw_hq_audio
                assembled["raw_lq_audio"] = raw_lq_audio
                assembled["raw_audio_sr"] = raw_audio_sr
            else:
                dummy_audio = torch.zeros(b, 8, 1, 16, device=device, dtype=torch.bfloat16)
                assembled["audio_hq_latents"] = {"latents": dummy_audio}
                assembled["audio_lq_latents"] = {"latents": dummy_audio}

        return original_training_step(self, assembled)

    import types
    trainer._training_step = types.MethodType(_online_training_step, trainer)


# ---------------------------------------------------------------------------
# Collate function (reuse from v2)
# ---------------------------------------------------------------------------

def online_collate_fn(samples: list[dict]) -> dict:
    """Collate OnlineAVDataset samples into a batch."""
    batch: dict = {
        "hq_video": torch.stack([s["hq_video"] for s in samples]),
        "lq_video": torch.stack([s["lq_video"] for s in samples]),
        "fps": torch.tensor([s["fps"] for s in samples]),
        "audio_sr": samples[0]["audio_sr"],
        "caption": [s.get("caption", "") for s in samples],
    }

    hq_audios = [s["hq_audio"] for s in samples]
    lq_audios = [s["lq_audio"] for s in samples]

    if all(a is not None for a in hq_audios):
        # Normalize to stereo (2ch) — mono [1, L] gets duplicated to [2, L]
        hq_audios = [a if a.shape[0] == 2 else a.expand(2, -1) for a in hq_audios]
        lq_audios = [a if a.shape[0] == 2 else a.expand(2, -1) for a in lq_audios]
        max_len = max(a.shape[-1] for a in hq_audios)
        batch["hq_audio"] = torch.stack([
            torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in hq_audios
        ])
        batch["lq_audio"] = torch.stack([
            torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in lq_audios
        ])
    else:
        batch["hq_audio"] = None
        batch["lq_audio"] = None

    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_online_v3.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    patch_strategy_registry()

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.model_loader import load_audio_vae_encoder, load_video_vae_encoder
    from ltx_trainer.trainer import LtxvTrainer

    # --- Determine training resolutions ---
    from ltx_core.types import SpatioTemporalScaleFactors
    scale = SpatioTemporalScaleFactors.default()

    data_cfg = raw_config.get("data", {})
    hq_res = data_cfg.get("hq_resolution")
    lq_res = data_cfg.get("lq_resolution")

    if hq_res and lq_res:
        hq_w, hq_h, max_f = hq_res
        lq_w, lq_h, _ = lq_res
    else:
        train_res = data_cfg.get("train_resolution")
        if train_res:
            hq_w, hq_h, max_f = train_res
        else:
            val_dims = raw_config.get("validation", {}).get("video_dims", [960, 576, 121])
            hq_w, hq_h, max_f = val_dims
        lq_w, lq_h = hq_w, hq_h

    hq_h = (hq_h // scale.height) * scale.height
    hq_w = (hq_w // scale.width) * scale.width
    lq_h = (lq_h // scale.height) * scale.height
    lq_w = (lq_w // scale.width) * scale.width
    logger.info(f"Super-resolution: LQ {lq_w}x{lq_h} → HQ {hq_w}x{hq_h}, frames={max_f}")

    # --- Setup fixed embeddings (encode once, unload Gemma) ---
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    embeddings_processor, cached_cond_feats, cached_empty_feats, cached_val_embeds = setup_fixed_embeddings(
        model_path=raw_config["model"]["model_path"],
        text_encoder_path=raw_config["model"]["text_encoder_path"],
        device=device,
    )

    # --- Create OnlineAVDataset ---
    from ltx_av_sr_trainer.online_av_dataset import OnlineAVDataset

    data_root = raw_config.get("data", {}).get("preprocessed_data_root", "")
    with_audio = raw_config.get("training_strategy", {}).get("with_audio", True)

    online_dataset = OnlineAVDataset(
        data_root=data_root,
        hq_width=hq_w,
        hq_height=hq_h,
        lq_width=lq_w,
        lq_height=lq_h,
        target_frames=max_f,
        with_audio=with_audio,
        seed=raw_config.get("seed", 42),
    )
    logger.info(f"OnlineAVDataset: {len(online_dataset)} samples from {data_root}")

    # --- Timestamp output_dir (multi-node safe via dist.broadcast) ---
    if "output_dir" in raw_config:
        import torch.distributed as dist

        num_gpus = int(os.environ.get("WORLD_SIZE", 1))

        if num_gpus > 1 and not dist.is_initialized():
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl")

        if rank == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Rank 0 generated timestamp: {ts}")
        else:
            ts = None

        if num_gpus > 1 and dist.is_initialized():
            if rank == 0:
                ts_bytes = ts.encode("utf-8")
                ts_tensor = torch.tensor(list(ts_bytes), dtype=torch.uint8, device=device)
            else:
                ts_tensor = torch.zeros(15, dtype=torch.uint8, device=device)
            dist.broadcast(ts_tensor, src=0)
            ts = bytes(ts_tensor.cpu().tolist()).decode("utf-8")

        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_config["output_dir"] = f"{raw_config['output_dir']}_{ts}_gpus{num_gpus}"

    # --- Build trainer config ---
    lora_structure_from = raw_config.get("model", {}).get("lora_structure_from")
    trainer_raw = {k: v for k, v in raw_config.items() if k not in ("swanlab", "logging")}

    trainer_raw.setdefault("data", {})["preprocessed_data_root"] = data_root
    trainer_raw.get("data", {}).pop("preprocess_resolution", None)
    trainer_raw.get("data", {}).pop("train_resolution", None)
    trainer_raw.get("data", {}).pop("hq_resolution", None)
    trainer_raw.get("data", {}).pop("lq_resolution", None)
    trainer_raw.get("model", {}).pop("lora_structure_from", None)

    av_strategy_raw = trainer_raw.get("training_strategy", {})
    if av_strategy_raw.get("name") == "av_restoration":
        trainer_raw["training_strategy"] = {
            "name": "text_to_video",
            "with_audio": av_strategy_raw.get("with_audio", True),
            "first_frame_conditioning_p": av_strategy_raw.get("first_frame_conditioning_p", 0.0),
        }

    trainer_config = LtxTrainerConfig(**trainer_raw)

    if av_strategy_raw.get("name") == "av_restoration":
        from ltx_av_sr_trainer.strategy import AVRestorationConfig
        trainer_config.training_strategy = AVRestorationConfig(**av_strategy_raw)

    # --- Apply SRTrainer patches ---
    sr_spatial = None
    if lq_w != hq_w or lq_h != hq_h:
        sr_spatial = {
            "lq_h": lq_h // scale.height,
            "lq_w": lq_w // scale.width,
            "hq_h": hq_h // scale.height,
            "hq_w": hq_w // scale.width,
        }
    SRTrainer.apply(LtxvTrainer, sr_spatial=sr_spatial, lora_structure_from=lora_structure_from)

    # --- Patch trainer internals (lightweight: no Gemma, no feature_extractor) ---
    _original_load_models = LtxvTrainer._load_models
    _original_load_text = LtxvTrainer._load_text_encoder_and_cache_embeddings
    _original_prepare = LtxvTrainer._prepare_models_for_training

    def _lightweight_load_text(self_trainer):
        self_trainer._embeddings_processor = embeddings_processor
        return [cached_val_embeds]

    def _lightweight_load_models(self_trainer):
        from ltx_trainer.model_loader import load_model as load_ltx_model
        components = load_ltx_model(
            checkpoint_path=self_trainer._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=True,
            with_audio_vae_decoder=True,
            with_vocoder=True,
            with_text_encoder=False,
        )
        self_trainer._transformer = components.transformer.to(dtype=torch.bfloat16)
        self_trainer._vae_decoder = components.video_vae_decoder
        self_trainer._vae_encoder = None
        self_trainer._scheduler = components.scheduler
        self_trainer._audio_vae = components.audio_vae_decoder
        self_trainer._vocoder = components.vocoder
        self_trainer._transformer.requires_grad_(False)
        if self_trainer._vae_decoder is not None:
            self_trainer._vae_decoder.requires_grad_(False).eval()
        if self_trainer._audio_vae is not None:
            self_trainer._audio_vae.requires_grad_(False).eval()
        if self_trainer._vocoder is not None:
            self_trainer._vocoder.requires_grad_(False).eval()
        del components
        gc.collect()

    def _lightweight_prepare(self_trainer):
        from accelerate import DistributedType
        if self_trainer._accelerator.distributed_type == DistributedType.FSDP and self_trainer._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self_trainer._transformer = self_trainer._transformer.to(dtype=torch.float32)
        transformer = (
            self_trainer._transformer.get_base_model()
            if hasattr(self_trainer._transformer, "get_base_model")
            else self_trainer._transformer
        )
        transformer.set_gradient_checkpointing(self_trainer._config.optimization.enable_gradient_checkpointing)
        if self_trainer._vae_decoder is not None:
            self_trainer._vae_decoder = self_trainer._vae_decoder.to("cpu")
        if self_trainer._vae_encoder is not None:
            self_trainer._vae_encoder = self_trainer._vae_encoder.to("cpu")
        self_trainer._transformer = self_trainer._accelerator.prepare(self_trainer._transformer)

    LtxvTrainer._load_text_encoder_and_cache_embeddings = _lightweight_load_text
    LtxvTrainer._load_models = _lightweight_load_models
    LtxvTrainer._prepare_models_for_training = _lightweight_prepare

    trainer = LtxvTrainer(trainer_config)

    LtxvTrainer._load_models = _original_load_models
    LtxvTrainer._load_text_encoder_and_cache_embeddings = _original_load_text
    LtxvTrainer._prepare_models_for_training = _original_prepare

    # Replace dataset
    trainer._dataset = online_dataset

    # Override _init_dataloader with DistributedSampler
    def _patched_init_dataloader(self=trainer):
        from torch.utils.data import DataLoader, DistributedSampler
        num_workers = self._config.data.num_dataloader_workers

        sampler = None
        shuffle = True
        if self._accelerator.num_processes > 1:
            sampler = DistributedSampler(
                self._dataset,
                num_replicas=self._accelerator.num_processes,
                rank=self._accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False
            logger.info(
                f"DistributedSampler: rank={self._accelerator.process_index}/"
                f"{self._accelerator.num_processes}, {len(self._dataset)} samples"
            )

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            collate_fn=online_collate_fn,
        )
        self._dataloader = dataloader

    trainer._init_dataloader = _patched_init_dataloader

    trainer._embeddings_processor = embeddings_processor

    # --- Logging and validation patches ---
    patch_logging(trainer, raw_config)
    _ensure_vae_encoder(trainer)

    # --- Cross-attention gradient isolation (if configured) ---
    isolation_layer = av_strategy_raw.get("cross_attn_grad_isolation_layer")
    if isolation_layer is not None:
        patch_cross_attention_grad_isolation(trainer, cutoff_layer=int(isolation_layer))

    val_data_root = os.environ.get("VAL_DATA_ROOT", data_root)
    lq_dims = (lq_w, lq_h, max_f) if lq_w != hq_w or lq_h != hq_h else None
    patch_validation(trainer, val_data_root, lq_dims=lq_dims)

    # --- Load VAE encoder on GPU ---
    model_path = raw_config["model"]["model_path"]
    vae_encoder = load_video_vae_encoder(model_path, device=device, dtype=torch.bfloat16)
    vae_encoder.requires_grad_(False).eval()
    logger.info("Video VAE encoder loaded (GPU)")

    audio_vae_encoder = None
    if with_audio:
        audio_vae_encoder = load_audio_vae_encoder(model_path, device=device, dtype=torch.bfloat16)
        audio_vae_encoder.requires_grad_(False).eval()
        logger.info("Audio VAE encoder loaded (GPU)")

    # --- Monkey-patch _training_step with fixed prompt ---
    _make_online_training_step_v3(
        trainer=trainer,
        vae_encoder=vae_encoder,
        audio_vae_encoder=audio_vae_encoder,
        cached_cond_feats=cached_cond_feats,
        cached_empty_feats=cached_empty_feats,
        with_audio=with_audio,
        rank=rank,
        world_size=world_size,
    )

    logger.info("v3 fixed-prompt mode: SR prompt pre-encoded, text encoder unloaded")

    # --- Train! ---
    trainer.train()


if __name__ == "__main__":
    main()
