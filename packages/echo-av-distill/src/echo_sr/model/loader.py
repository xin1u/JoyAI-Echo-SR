"""Model loading utilities for Echo SR trainer."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from echo_sr import logger


def load_transformer(model_path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """Load LTX-2.3 transformer only (skip decoder/vocoder for faster startup)."""
    from ltx_trainer.model_loader import load_transformer as _load_transformer
    transformer = _load_transformer(model_path, device=device, dtype=dtype)
    transformer.requires_grad_(False)
    logger.info(f"Transformer loaded: {model_path}")
    return transformer


def load_vae_encoder(model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load video VAE encoder on GPU."""
    from ltx_trainer.model_loader import load_video_vae_encoder
    enc = load_video_vae_encoder(model_path, device=device, dtype=dtype)
    enc.requires_grad_(False).eval()
    logger.info("Video VAE encoder loaded (GPU)")
    return enc


def load_audio_vae_encoder(model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load audio VAE encoder on GPU."""
    from ltx_trainer.model_loader import load_audio_vae_encoder as _load
    enc = _load(model_path, device=device, dtype=dtype)
    enc.requires_grad_(False).eval()
    logger.info("Audio VAE encoder loaded (GPU)")
    return enc


def load_video_vae_decoder(model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load video VAE decoder on GPU."""
    from ltx_trainer.model_loader import load_video_vae_decoder as _load
    dec = _load(model_path, device=device, dtype=dtype)
    dec.requires_grad_(False).eval()
    logger.info("Video VAE decoder loaded (GPU)")
    return dec


def load_embeddings_processor(model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load embeddings processor (connectors only, no feature extractor)."""
    from ltx_trainer.model_loader import load_embeddings_processor as _load
    proc = _load(model_path, device=device, dtype=dtype)
    proc.requires_grad_(False).eval()
    return proc


def load_text_encoder(text_encoder_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load Gemma text encoder."""
    from ltx_trainer.model_loader import load_text_encoder as _load
    enc = _load(gemma_model_path=text_encoder_path, device=device, dtype=dtype, load_in_8bit=False)
    enc.requires_grad_(False).eval()
    logger.info(f"Text encoder loaded: {text_encoder_path}")
    return enc


def init_cond_proj(transformer, sr_spatial: dict | None = None, cond_proj_cfg: dict | None = None, init_std: float = 1e-6) -> list[nn.Parameter]:
    """Initialize condition projection layers for SR injection.

    Args:
        transformer: The LTX transformer model.
        sr_spatial: Dict with lq_h, lq_w, hq_h, hq_w for fixed-resolution SR.
        cond_proj_cfg: Config dict for condition projector type. If type="latent_2x", uses CondLatent2xProj.
        init_std: Initialization std for projection layer.

    Returns list of trainable cond_proj parameters.
    """
    base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer

    proj_type = (cond_proj_cfg or {}).get("type", "default")

    if proj_type == "latent_2x":
        # Resolution-independent 2x latent upsampling
        from echo_sr.model.cond_latent_2x import CondLatent2xProj
        inner_dim = 4096  # LTX-2.3 video hidden dim
        mid_channels = (cond_proj_cfg or {}).get("mid_channels", 128)
        proj = CondLatent2xProj(in_channels=128, inner_dim=inner_dim, mid_channels=mid_channels)
        CondLatent2xProj.init_near_zero(proj, proj_std=init_std)
        # Register as _cond_sr_video_proj so existing preprocessor interface works
        base._cond_sr_video_proj = proj
        # Audio proj
        audio_proj = nn.Linear(128, 2048)
        nn.init.normal_(audio_proj.weight, std=init_std)
        nn.init.zeros_(audio_proj.bias)
        base._cond_audio_proj = audio_proj
        base._rebuild_preprocessors_with_cond(use_sr_proj=True)
    elif proj_type == "pixel_wave":
        # Pixel + Waveform direct condition injection (no VAE encode for LQ)
        from echo_sr.model.pixel_cond_proj import PixelCondProj
        from echo_sr.model.wave_cond_proj import WaveCondProj
        cfg = cond_proj_cfg or {}
        video_proj = PixelCondProj(
            in_channels=3,
            inner_dim=4096,
            hidden_dim1=cfg.get("hidden_dim1", 512),
            hidden_dim2=cfg.get("hidden_dim2", 768),
            hidden_dim3=cfg.get("hidden_dim3", 1024),
        )
        PixelCondProj.init_near_zero(video_proj, std=init_std)
        audio_proj = WaveCondProj(
            audio_inner_dim=2048,
            source_sr=cfg.get("audio_source_sr", 44100),
            target_sr=cfg.get("audio_target_sr", 16000),
        )
        WaveCondProj.init_near_zero(audio_proj, std=init_std)
        base._cond_sr_video_proj = video_proj
        base._cond_audio_proj = audio_proj
        base._rebuild_preprocessors_with_cond(use_sr_proj=True)
    elif sr_spatial:
        base.init_cond_sr_proj(**sr_spatial, init_std=init_std)
    else:
        base.init_cond_proj(init_std=init_std)

    cond_params = []
    for name, param in base.named_parameters():
        if ("_cond_" in name or name.startswith("cond_")) and "proj" in name:
            param.requires_grad = True
            cond_params.append(param)
            logger.info(f"  cond_proj trainable: {name} {list(param.shape)}")

    logger.info(f"Cond proj initialized (type={proj_type}): {sum(p.numel() for p in cond_params):,} params")
    return cond_params


def encode_fixed_prompts(
    model_path: str,
    text_encoder_path: str,
    device: torch.device,
    sr_prompt: str,
    negative_prompt: str,
    cache_path: Path | None = None,
) -> tuple[dict, object]:
    """Encode fixed SR prompt. Cache to disk for reuse across runs.

    Returns (cond_feats, val_embeds).
    No empty/negative embeddings — fixed prompt SR doesn't use text CFG.
    """
    import gc
    from ltx_trainer.validation_sampler import CachedPromptEmbeddings

    # Try cache
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("sr_prompt_text") == sr_prompt:
            if "cond_embeds" in cached:
                logger.info(f"Prompt embeddings loaded from cache: {cache_path}")
                val_embeds = CachedPromptEmbeddings(
                    video_context_positive=cached["val_positive"]["video_encoding"],
                    audio_context_positive=cached["val_positive"]["audio_encoding"],
                    video_context_negative=None,
                    audio_context_negative=None,
                )
                return cached["cond_embeds"], val_embeds
            else:
                logger.info("Cache format outdated, re-encoding...")

    # Encode from scratch — run full Block 1+2+3 pipeline
    text_enc = load_text_encoder(text_encoder_path, device=device)
    proc = load_embeddings_processor(model_path, device=device)

    with torch.no_grad():
        sr_hs, sr_mask = text_enc.encode(sr_prompt, padding_side="left")
        pos_out = proc.process_hidden_states(sr_hs, sr_mask, padding_side="left")

    val_embeds = CachedPromptEmbeddings(
        video_context_positive=pos_out.video_encoding.cpu(),
        audio_context_positive=pos_out.audio_encoding.cpu(),
        video_context_negative=None,
        audio_context_negative=None,
    )

    cond_embeds = {
        "video_embeds": pos_out.video_encoding.cpu(),
        "audio_embeds": pos_out.audio_encoding.cpu() if pos_out.audio_encoding is not None else None,
        "attention_mask": pos_out.attention_mask.cpu(),
    }

    # Unload
    del text_enc
    proc.feature_extractor = None
    gc.collect()
    torch.cuda.empty_cache()

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cond_embeds": cond_embeds,
            "val_positive": {"video_encoding": val_embeds.video_context_positive, "audio_encoding": val_embeds.audio_context_positive},
            "sr_prompt_text": sr_prompt,
        }, cache_path)
        logger.info(f"Prompt embeddings cached: {cache_path}")

    return cond_embeds, val_embeds
