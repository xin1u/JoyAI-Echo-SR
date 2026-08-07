#!/usr/bin/env python3
"""Long-video TAV2AV inference: sliding window + drop-first-frame i2v chaining.

Loads the 1-step distilled LoRA, merges it into the dev base model, and runs
sliding-window inference.

Every window after the first is turned into an i2v problem whose conditioning
frame is then thrown away:

  1. the previous window's LAST frame latent is written into this window's first
     H*W token slots (both `latent` and `clean_latent`) and `denoise_mask` is
     zeroed there, so the solver treats those tokens as given;
  2. after each Euler step the prediction for those tokens is overwritten with
     the conditioning latent again, so a 1-step solver cannot drift off it;
  3. `clear_conditioning()` drops them before unpatchify/decode -- the frame is a
     duplicate of one the previous window already emitted, so each window
     contributes 120 new frames and the concatenation is continuous.

The first window of a shot has no predecessor and runs in plain t2av mode.
`first_frame_conditioning_p` in the training configs is the training-time
counterpart of the same formulation.

Usage (single GPU):
  python scripts/infer_distill_v3_long.py

Usage (multi-GPU — windows distributed across ranks):
  torchrun --nproc_per_node=8 scripts/infer_distill_v3_long.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
_MONO_ROOT = _PKG_ROOT.parent
sys.path.insert(0, str(_PKG_ROOT / "src"))
sys.path.insert(0, str(_MONO_ROOT / "ltx-trainer-1.1" / "src"))
sys.path.insert(0, str(_MONO_ROOT / "ltx-core-1.1" / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "checkpoints/ltx-2.3-22b-dev.safetensors"
CHECKPOINT_PATH = "checkpoints/echo-sr/av-sr-1k-distill-video-step005100.safetensors"

INPUT_VIDEO = ""  # required CLI arg

LQ_W, LQ_H = 1280, 736
HQ_W, HQ_H = 1920, 1152
WINDOW_FRAMES = 121
WINDOW_STRIDE = 97  # only feeds the crossfade ramp width; see compute_shot_windows
                    # for the real schedule (1-frame overlap inside a shot)
FPS_MODEL = 25.0
FRAMES_PER_SHOT = 241

INFERENCE_STEPS = 1  # distilled 1-step
GUIDANCE_SCALE = 1.0  # distilled model doesn't need CFG
STG_SCALE = 0.0
STG_BLOCKS = []
SEED = 42
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0")) or None

OUTPUT_DIR = _PKG_ROOT / "outputs" / "distill_v3_long_inference"

AUDIO_SAMPLE_RATE = 44100

PROMPT_CACHE_PATH = Path(
    os.environ.get("AV_SR_PROMPT_CACHE", "checkpoints/prompt/sr_prompt_embeddings.pt")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def read_video_window(path: str, start_frame: int, num_frames: int) -> torch.Tensor:
    """Read [C, F, H, W] float32 in [-1, 1]."""
    container = av.open(path)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i < start_frame:
            continue
        if i >= start_frame + num_frames:
            break
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    video = torch.stack(frames)
    video = video.permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return video


def read_audio_window(path: str, start_sec: float, duration_sec: float) -> tuple[torch.Tensor, int]:
    """Read [C, T] float32 waveform."""
    container = av.open(path)
    if len(container.streams.audio) == 0:
        container.close()
        return torch.zeros(2, 0), AUDIO_SAMPLE_RATE

    astream = container.streams.audio[0]
    sr = astream.rate
    audio_frames = []
    for aframe in container.decode(audio=0):
        arr = aframe.to_ndarray()
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        audio_frames.append(torch.from_numpy(arr.copy()))
    container.close()

    if not audio_frames:
        return torch.zeros(2, 0), sr
    audio = torch.cat(audio_frames, dim=1).float()
    if audio.shape[0] == 1:
        audio = audio.expand(2, -1)

    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    audio = audio[:, start_sample:end_sample]
    return audio, sr


def get_video_info(path: str) -> tuple[int, int, int]:
    container = av.open(path)
    vstream = container.streams.video[0]
    fps = int(round(float(vstream.average_rate)))
    total_frames = int(vstream.frames)
    if total_frames == 0:
        total_frames = sum(1 for _ in container.decode(video=0))
        container.close()
        container = av.open(path)
    audio_sr = 48000
    if len(container.streams.audio) > 0:
        audio_sr = container.streams.audio[0].rate
    container.close()
    return total_frames, fps, audio_sr


def compute_shot_windows(num_shots: int) -> list[tuple[int, int, int]]:
    """(shot_idx, start_frame, end_frame), 2 windows per shot."""
    windows = []
    for shot_idx in range(num_shots):
        shot_start = shot_idx * FRAMES_PER_SHOT
        windows.append((shot_idx, shot_start, shot_start + 121))
        windows.append((shot_idx, shot_start + 120, shot_start + 241))
    return windows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: torch.device, rank: int):
    from ltx_trainer.model_loader import (
        load_audio_vae_decoder,
        load_audio_vae_encoder,
        load_transformer,
        load_video_vae_encoder,
        load_vocoder,
    )

    log(rank, "  Loading transformer...")
    transformer = load_transformer(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    transformer.eval()

    lq_h, lq_w = LQ_H // 32, LQ_W // 32
    hq_h, hq_w = HQ_H // 32, HQ_W // 32
    log(rank, f"  Init CondSRPatchifyProj: LQ {lq_h}x{lq_w} -> HQ {hq_h}x{hq_w}")
    transformer.init_cond_sr_proj(lq_h=lq_h, lq_w=lq_w, hq_h=hq_h, hq_w=hq_w)

    # LoRA: load distilled checkpoint and merge
    log(rank, "  Applying distilled LoRA...")
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    ckpt_sd = load_file(CHECKPOINT_PATH)
    lora_keys = [k for k in ckpt_sd if "lora_A" in k and "cond_" not in k]
    target_modules: set[str] = set()
    ranks: dict[str, int] = {}
    for k in lora_keys:
        parts = k.replace("diffusion_model.", "").split(".")
        lora_idx = next(i for i, p in enumerate(parts) if p == "lora_A")
        module_name = ".".join(parts[:lora_idx])
        target_modules.add(module_name)
        ranks[module_name] = ckpt_sd[k].shape[0]

    max_rank = max(ranks.values())
    rank_pattern = {k: v for k, v in ranks.items() if v != max_rank}
    alpha_pattern = {k: v for k, v in ranks.items() if v != max_rank}

    lora_config = LoraConfig(
        r=max_rank,
        lora_alpha=max_rank,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    transformer = get_peft_model(transformer, lora_config)

    remapped = {}
    for k, v in ckpt_sd.items():
        k = k.replace("diffusion_model.", "", 1)
        remapped[f"base_model.model.{k}"] = v
    del ckpt_sd

    cond_keys = {k for k in remapped if "cond_" in k and "proj" in k}
    cond_state = {k: remapped[k] for k in cond_keys}
    lora_state = {k: v for k, v in remapped.items() if k not in cond_keys}

    set_peft_model_state_dict(transformer, lora_state)

    if cond_state:
        base = transformer.get_base_model()
        for name, param in base.named_parameters():
            full_key = f"base_model.model.{name}"
            if full_key in cond_state:
                param.data.copy_(cond_state[full_key].to(param.dtype))
    del remapped, lora_state, cond_state

    log(rank, "  Merging LoRA into base weights...")
    transformer = transformer.merge_and_unload()
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()
    log(rank, f"  GPU mem after transformer: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    log(rank, "  Loading video VAE encoder...")
    vae_encoder = load_video_vae_encoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    vae_encoder.eval()

    log(rank, "  Loading TinyDecoder (taeltx2_3_wide)...")
    from ltx_av_sr_trainer.tiny_decoder import TAEHV
    TINY_DECODER_PATH = str(Path(__file__).resolve().parent.parent / "ckpt" / "taeltx2_3_wide.pth")
    tiny_decoder = TAEHV(checkpoint_path=TINY_DECODER_PATH).to(device=device, dtype=torch.bfloat16)
    tiny_decoder.eval().requires_grad_(False)

    log(rank, "  Loading audio VAE + vocoder...")
    audio_encoder = load_audio_vae_encoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    audio_encoder.eval()
    audio_decoder = load_audio_vae_decoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    audio_decoder.eval()
    vocoder = load_vocoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    vocoder.eval()

    return {
        "transformer": transformer,
        "vae_encoder": vae_encoder,
        "tiny_decoder": tiny_decoder,
        "audio_encoder": audio_encoder,
        "audio_decoder": audio_decoder,
        "vocoder": vocoder,
    }


def load_cached_embeddings(device: torch.device, rank: int):
    """Load pre-cached fixed prompt embeddings — no text encoder needed."""
    log(rank, f"  Loading cached embeddings from {PROMPT_CACHE_PATH}")
    cached = torch.load(PROMPT_CACHE_PATH, map_location="cpu", weights_only=False)

    v_pos = cached["val_positive"]["video_encoding"].to(device)
    a_pos = cached["val_positive"]["audio_encoding"]
    a_pos = a_pos.to(device) if a_pos is not None else None
    v_neg = cached["val_negative"]["video_encoding"].to(device)
    a_neg = cached["val_negative"]["audio_encoding"]
    a_neg = a_neg.to(device) if a_neg is not None else None

    log(rank, f"  v_pos: {list(v_pos.shape)}, a_pos: {list(a_pos.shape) if a_pos is not None else None}")
    return v_pos, a_pos, v_neg, a_neg


def build_sigma_schedule(num_frames: int, device: torch.device) -> torch.Tensor:
    if INFERENCE_STEPS == 1:
        # Distilled 1-step model: trained with sigma=1.0 → x0, must match training validation
        return torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)

    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.types import VideoLatentShape, VideoPixelShape

    pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=HQ_H, width=HQ_W, fps=FPS_MODEL)
    target_shape = VideoLatentShape.from_pixel_shape(pixel_shape)

    scheduler = LTX2Scheduler()
    tokens = target_shape.frames * target_shape.height * target_shape.width
    SHIFT_CAP = 13.0
    _x1, _x2, _base, _max_default = 1024, 4096, 0.95, 2.05
    raw_shift = (_max_default - _base) * (tokens - _x1) / (_x2 - _x1) + _base
    if raw_shift > SHIFT_CAP:
        safe_max = _base + (SHIFT_CAP - _base) * (_x2 - _x1) / (tokens - _x1)
    else:
        safe_max = _max_default

    dummy = torch.empty(1, 1, target_shape.frames, target_shape.height, target_shape.width, device=device)
    sigmas = scheduler.execute(steps=INFERENCE_STEPS, latent=dummy, max_shift=safe_max).to(device).float()
    return sigmas


def generate_global_noise(total_frames: int, device: torch.device) -> torch.Tensor:
    from ltx_core.types import SpatioTemporalScaleFactors

    sf = SpatioTemporalScaleFactors.default()
    t_latent = (total_frames - 1) // sf.time + 1
    h_latent = HQ_H // sf.height
    w_latent = HQ_W // sf.width

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn(1, 128, t_latent, h_latent, w_latent, generator=generator)
    return noise


# ---------------------------------------------------------------------------
# Core window processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_window(
    video_chunk: torch.Tensor,
    audio_waveform: torch.Tensor,
    audio_sr: int,
    models: dict,
    v_pos: torch.Tensor,
    a_pos: torch.Tensor | None,
    v_neg: torch.Tensor,
    a_neg: torch.Tensor | None,
    window_idx: int,
    device: torch.device,
    global_noise: torch.Tensor,
    frame_start: int,
    src_fps: int,
    prev_last_frame_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Process a single window with optional first-frame conditioning.

    Args:
        prev_last_frame_latent: If provided, the patchified latent of the last
            frame from the previous window. Used to set the first frame as
            clean conditioning (v2v mode) via denoise_mask=0.

    Returns:
        hq_video: [C, F, H_hq, W_hq] float in [0, 1]
        hq_audio: [C, T] float waveform or None
        last_frame_latent: patchified latent of the last output frame for next window
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.model.audio_vae import AudioProcessor, encode_audio
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import (
        Audio,
        AudioLatentShape,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
        VideoPixelShape,
    )

    num_frames = video_chunk.shape[1]
    VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    # --- Video: pixel -> latent ---
    video_pixel = video_chunk.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    models["vae_encoder"].to(device)
    lq_latent = models["vae_encoder"].tiled_encode(video_pixel, tiling_config=TilingConfig.default())
    models["vae_encoder"].to("cpu")
    torch.cuda.empty_cache()

    # --- Audio: waveform -> latent ---
    lq_audio_tokens = None
    window_duration = float(num_frames) / float(src_fps)

    if audio_waveform is not None and audio_waveform.shape[1] > 0:
        models["audio_encoder"].to(device)
        enc = models["audio_encoder"]
        audio_processor = AudioProcessor(
            target_sample_rate=enc.sample_rate,
            mel_bins=enc.mel_bins,
            mel_hop_length=enc.mel_hop_length,
            n_fft=enc.n_fft,
        ).to(device=device)

        wf = audio_waveform
        if wf.dim() == 1:
            wf = wf.unsqueeze(0)
        if wf.shape[0] == 1:
            wf = wf.expand(2, -1)

        audio_obj = Audio(
            waveform=wf.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
            sampling_rate=audio_sr,
        )
        audio_latent = encode_audio(audio_obj, enc, audio_processor)
        lq_audio_tokens = audio_patchifier.patchify(audio_latent)
        models["audio_encoder"].to("cpu")
        del audio_processor
        torch.cuda.empty_cache()

    # --- Build tools ---
    pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=HQ_H, width=HQ_W, fps=FPS_MODEL)
    target_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools = VideoLatentTools(
        patchifier=video_patchifier,
        target_shape=target_shape,
        fps=FPS_MODEL,
        scale_factors=VIDEO_SCALE_FACTORS,
        causal_fix=True,
    )

    audio_tools = None
    if lq_audio_tokens is not None:
        audio_tools = AudioLatentTools(
            patchifier=audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=window_duration),
        )

    # --- Sigma schedule ---
    sigmas = build_sigma_schedule(num_frames, device)
    stepper = EulerDiffusionStep()
    x0_model = X0Model(models["transformer"])

    # --- Video: slice global noise for this window ---
    sf = VIDEO_SCALE_FACTORS
    t_start_latent = frame_start // sf.time
    t_frames_latent = target_shape.frames
    window_noise = global_noise[
        :, :,
        t_start_latent : t_start_latent + t_frames_latent,
        :target_shape.height,
        :target_shape.width,
    ].clone().to(device=device, dtype=torch.bfloat16)

    if window_noise.shape[2] < t_frames_latent:
        pad_t = t_frames_latent - window_noise.shape[2]
        pad_noise = torch.randn(
            1, 128, pad_t, target_shape.height, target_shape.width,
            device=device, dtype=torch.bfloat16,
        )
        window_noise = torch.cat([window_noise, pad_noise], dim=2)

    # --- Create initial states ---
    video_state = video_tools.create_initial_state(device, torch.bfloat16)
    patchified_noise = video_patchifier.patchify(window_noise)
    video_state = replace(video_state, latent=patchified_noise)

    # --- V2V: apply first-frame conditioning from previous window ---
    num_first_frame_tokens = target_shape.height * target_shape.width
    if prev_last_frame_latent is not None:
        new_latent = video_state.latent.clone()
        new_latent[:, :num_first_frame_tokens] = prev_last_frame_latent.to(new_latent.dtype)

        new_clean_latent = video_state.clean_latent.clone()
        new_clean_latent[:, :num_first_frame_tokens] = prev_last_frame_latent.to(new_clean_latent.dtype)

        new_denoise_mask = video_state.denoise_mask.clone()
        new_denoise_mask[:, :num_first_frame_tokens] = 0.0

        video_state = replace(
            video_state,
            latent=new_latent,
            clean_latent=new_clean_latent,
            denoise_mask=new_denoise_mask,
        )

    audio_state = None
    if audio_tools is not None:
        from ltx_core.components.noisers import GaussianNoiser
        audio_state = audio_tools.create_initial_state(device, torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(SEED + window_idx)
        noiser = GaussianNoiser(generator)
        audio_state = noiser(audio_state, noise_scale=1.0)

    # --- Align tokens helper ---
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

    # --- Build Modality objects ---
    lq_video_cond = lq_latent

    video_mod = Modality(
        enabled=True,
        latent=video_state.latent,
        sigma=sigmas[0].repeat(video_state.latent.shape[0]),
        timesteps=video_state.denoise_mask * sigmas[0],
        positions=video_state.positions,
        context=v_pos,
        context_mask=None,
        cond_latent=lq_video_cond,
    )

    audio_mod = None
    if audio_state is not None and lq_audio_tokens is not None:
        aligned_lq_audio = _align_tokens(lq_audio_tokens, audio_state.latent.shape[1])
        audio_mod = Modality(
            enabled=True,
            latent=audio_state.latent,
            sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
            timesteps=audio_state.denoise_mask * sigmas[0],
            positions=audio_state.positions,
            context=a_pos,
            context_mask=None,
            cond_latent=aligned_lq_audio,
        )

    # --- Denoising loop (1-step for distilled model) ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

            denoised_video, denoised_audio = x0_model(
                video=video_mod, audio=audio_mod, perturbations=None
            )

            # For v2v: keep conditioned first-frame tokens unchanged
            if prev_last_frame_latent is not None:
                denoised_video[:, :num_first_frame_tokens] = prev_last_frame_latent.to(denoised_video.dtype)

            # Euler step
            video_state = replace(
                video_state,
                latent=stepper.step(
                    sample=video_mod.latent, denoised_sample=denoised_video,
                    sigmas=sigmas, step_index=step_idx,
                ),
            )
            if audio_mod is not None and audio_state is not None and denoised_audio is not None:
                audio_state = replace(
                    audio_state,
                    latent=stepper.step(
                        sample=audio_mod.latent, denoised_sample=denoised_audio,
                        sigmas=sigmas, step_index=step_idx,
                    ),
                )

    # --- Extract last-frame latent (patchified) for v2v chaining ---
    last_frame_latent = video_state.latent[:, -num_first_frame_tokens:]

    # --- Decode video via TinyDecoder ---
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)

    # TinyDecoder expects [B, T, C, H, W], outputs [B, T, 3, H_px, W_px] in [0,1]
    lat_ntchw = video_state.latent.unsqueeze(0) if video_state.latent.dim() == 4 else video_state.latent
    lat_ntchw = lat_ntchw.permute(0, 2, 1, 3, 4).to(torch.bfloat16)  # [B, C, F, H, W] → [B, F, C, H, W]
    vid_01 = models["tiny_decoder"].decode_video(lat_ntchw, parallel=True, show_progress_bar=False)
    # [B, F, 3, H, W] → [3, F, H, W] in [0,1]
    hq_video = vid_01[0].permute(1, 0, 2, 3).float().clamp(0, 1)

    # --- Decode audio ---
    hq_audio = None
    if audio_state is not None and audio_tools is not None:
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        models["audio_decoder"].to(device)
        audio_mel = models["audio_decoder"](audio_state.latent)
        models["audio_decoder"].to("cpu")
        torch.cuda.empty_cache()

        models["vocoder"].to(device)
        hq_audio = models["vocoder"](audio_mel).squeeze(0).float()
        models["vocoder"].to("cpu")
        torch.cuda.empty_cache()

    return hq_video, hq_audio, last_frame_latent


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def _make_crossfade_weights(n: int, ramp_len: int, is_first: bool, is_last: bool) -> torch.Tensor:
    w = torch.ones(n)
    if ramp_len <= 0:
        return w
    ramp = min(ramp_len, n // 2)
    if not is_first:
        w[:ramp] = torch.linspace(0.0, 1.0, ramp + 2)[1:-1]
    if not is_last:
        w[-ramp:] = torch.linspace(1.0, 0.0, ramp + 2)[1:-1]
    return w


def blend_video_windows(
    results: list[tuple[int, int, torch.Tensor]],
    total_frames: int,
) -> torch.Tensor:
    overlap = WINDOW_FRAMES - WINDOW_STRIDE
    C, _, H, W = results[0][2].shape
    output = torch.zeros(C, total_frames, H, W)
    weights = torch.zeros(1, total_frames, 1, 1)

    for i, (start, end, vid) in enumerate(results):
        n = vid.shape[1]
        w = _make_crossfade_weights(n, overlap, is_first=(i == 0), is_last=(i == len(results) - 1))
        w = w.view(1, n, 1, 1)
        output[:, start:start + n] += vid[:, :n] * w
        weights[:, start:start + n] += w

    weights = weights.clamp(min=1e-8)
    output = output / weights
    return output


def blend_audio_windows(
    results: list[tuple[float, float, torch.Tensor]],
    total_duration: float,
    sr: int,
) -> torch.Tensor:
    overlap_sec = float(WINDOW_FRAMES - WINDOW_STRIDE) / FPS_MODEL
    overlap_samples = int(overlap_sec * sr)

    total_samples = int(total_duration * sr)
    C = results[0][2].shape[0]
    output = torch.zeros(C, total_samples)
    weights = torch.zeros(1, total_samples)

    for i, (start_sec, end_sec, wav) in enumerate(results):
        start_sample = int(start_sec * sr)
        n_samples = wav.shape[1]
        end_sample = start_sample + n_samples
        if end_sample > total_samples:
            n_samples = total_samples - start_sample
            wav = wav[:, :n_samples]
            end_sample = total_samples
        w = _make_crossfade_weights(n_samples, overlap_samples, is_first=(i == 0), is_last=(i == len(results) - 1))
        w = w.unsqueeze(0)
        output[:, start_sample:end_sample] += wav * w
        weights[:, start_sample:end_sample] += w

    weights = weights.clamp(min=1e-8)
    output = output / weights
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Long-video AV SR inference (1-step distilled, sliding window)")
    p.add_argument("--input", required=True, help="Input LQ video (with audio track)")
    p.add_argument("--model-path", default=MODEL_PATH, help="Base model safetensors")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Distilled LoRA checkpoint")
    p.add_argument("--prompt-cache", default=str(PROMPT_CACHE_PATH), help="Cached SR prompt embeddings (.pt)")
    p.add_argument("--output-dir", default="outputs/distill_v3_long_inference")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    global MODEL_PATH, CHECKPOINT_PATH, INPUT_VIDEO, PROMPT_CACHE_PATH, OUTPUT_DIR, SEED

    args = parse_args()
    MODEL_PATH = args.model_path
    CHECKPOINT_PATH = args.checkpoint
    INPUT_VIDEO = args.input
    PROMPT_CACHE_PATH = Path(args.prompt_cache)
    OUTPUT_DIR = Path(args.output_dir)
    SEED = args.seed

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUTPUT_DIR / "tmp_windows"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log(rank, "=" * 70)
    log(rank, "Distilled TAV2AV Long Video Inference (1-step, v2v sliding window)")
    log(rank, f"  Rank {rank}/{world_size}, device={device}")
    log(rank, f"  Checkpoint: {CHECKPOINT_PATH}")
    log(rank, f"  Inference steps: {INFERENCE_STEPS}")
    log(rank, "=" * 70)

    # --- 1. Read input metadata ---
    log(rank, "\n[1/6] Reading input video metadata...")
    total_frames, src_fps, audio_sr = get_video_info(INPUT_VIDEO)
    total_duration = total_frames / src_fps
    log(rank, f"  Video: {total_frames} frames, fps={src_fps}, duration={total_duration:.1f}s")

    num_shots = total_frames // FRAMES_PER_SHOT
    log(rank, f"  Detected {num_shots} shots ({FRAMES_PER_SHOT} frames each)")

    # --- 2. Compute windows + global noise ---
    windows = compute_shot_windows(num_shots)
    if MAX_WINDOWS:
        windows = windows[:MAX_WINDOWS]
    log(rank, f"\n[2/6] {len(windows)} windows total")

    log(rank, "  Generating global noise field (CPU)...")
    global_noise = generate_global_noise(total_frames, device)
    log(rank, f"  Global noise: {list(global_noise.shape)}")

    # --- 3. Load models ---
    log(rank, f"\n[3/6] Loading models (rank {rank})...")
    models = load_models(device, rank)

    # --- 4. Load cached prompt embeddings (no text encoder needed) ---
    log(rank, "\n[4/6] Loading cached prompt embeddings...")
    v_pos, a_pos, v_neg, a_neg = load_cached_embeddings(device, rank)

    # --- 5. Process windows sequentially with v2v chaining ---
    log(rank, f"\n[5/6] Processing windows (1-step distilled, v2v sliding window)...")

    # For v2v chaining: track last-frame latent per shot
    shot_last_frame_latent: dict[int, torch.Tensor] = {}

    if distributed:
        # Group windows by shot so v2v chaining stays within one rank.
        # Each shot has 2 consecutive windows; assign whole shots to ranks.
        shots_in_order = sorted(set(s for s, _, _ in windows))
        my_shots = set(shots_in_order[i] for i in range(len(shots_in_order)) if i % world_size == rank)
        my_windows = [(i, w) for i, w in enumerate(windows) if w[0] in my_shots]
    else:
        my_windows = list(enumerate(windows))

    for wi, (win_idx, (shot_idx, start, end)) in enumerate(my_windows):
        n_frames = end - start
        start_sec = start / src_fps
        duration_sec = n_frames / src_fps

        # Determine if this is the first window of a shot
        shot_start = shot_idx * FRAMES_PER_SHOT
        is_first_window_of_shot = (start == shot_start)

        # v2v: use previous window's last frame as first-frame condition
        prev_latent = None
        if not is_first_window_of_shot and shot_idx in shot_last_frame_latent:
            prev_latent = shot_last_frame_latent[shot_idx]

        mode_str = "t2av" if prev_latent is None else "v2v"
        print(
            f"  [Rank {rank}] Window {wi+1}/{len(my_windows)} "
            f"(global {win_idx+1}/{len(windows)}): shot={shot_idx} "
            f"frames [{start}:{end}] ({n_frames}f) mode={mode_str}",
            flush=True,
        )

        video_chunk = read_video_window(INPUT_VIDEO, start, n_frames)
        audio_chunk, chunk_audio_sr = read_audio_window(INPUT_VIDEO, start_sec, duration_sec)

        hq_video, hq_audio, last_frame_latent = process_window(
            video_chunk=video_chunk,
            audio_waveform=audio_chunk,
            audio_sr=chunk_audio_sr,
            models=models,
            v_pos=v_pos, a_pos=a_pos,
            v_neg=v_neg, a_neg=a_neg,
            window_idx=win_idx,
            device=device,
            global_noise=global_noise,
            frame_start=start,
            src_fps=src_fps,
            prev_last_frame_latent=prev_latent,
        )

        # Save last-frame latent for v2v chaining
        shot_last_frame_latent[shot_idx] = last_frame_latent.cpu()

        save_data = {
            "video": (start, end, hq_video.cpu()),
            "audio_start_sec": start_sec,
            "audio_end_sec": start_sec + duration_sec,
        }
        if hq_audio is not None:
            save_data["audio"] = hq_audio.cpu()
        torch.save(save_data, tmp_dir / f"window_{start}_{end}.pt")

        print(
            f"  [Rank {rank}] Window {win_idx+1} done: "
            f"video={list(hq_video.shape)}, "
            f"audio={list(hq_audio.shape) if hq_audio is not None else None}, "
            f"GPU={torch.cuda.memory_allocated(device)/1e9:.1f}GB",
            flush=True,
        )

    # --- 6. Gather + blend + save ---
    if distributed:
        dist.barrier()

    if rank == 0:
        log(rank, f"\n[6/6] Blending per-shot and concatenating...")

        all_video_results = []
        all_audio_results = []
        for shot_idx, s, e in windows:
            wpath = tmp_dir / f"window_{s}_{e}.pt"
            if wpath.exists():
                data = torch.load(wpath, weights_only=False)
                v_start, v_end, v_tensor = data["video"]
                all_video_results.append((v_start, v_end, v_tensor))
                if "audio" in data:
                    all_audio_results.append((
                        data["audio_start_sec"],
                        data["audio_end_sec"],
                        data["audio"],
                    ))

        all_video_results.sort(key=lambda x: x[0])
        all_audio_results.sort(key=lambda x: x[0])

        covered_end = max(e for _, _, e in windows)
        blend_frames = min(total_frames, covered_end)

        final_video = blend_video_windows(all_video_results, blend_frames)
        log(rank, f"  Blended video: {list(final_video.shape)}")

        final_audio = None
        audio_out_sr = None
        if all_audio_results:
            audio_out_sr = int(getattr(models["vocoder"], "output_sampling_rate", 48000))
            blend_duration = blend_frames / src_fps
            final_audio = blend_audio_windows(all_audio_results, blend_duration, audio_out_sr)
            log(rank, f"  Blended audio: {list(final_audio.shape)}, sr={audio_out_sr}")

        from ltx_trainer.video_utils import save_video

        output_path = OUTPUT_DIR / "enhanced_distill_v3.mp4"
        save_video(
            video_tensor=final_video,
            output_path=output_path,
            fps=float(src_fps),
            audio=final_audio if final_audio is not None else None,
            audio_sample_rate=audio_out_sr if final_audio is not None else None,
        )

        log(rank, f"\n{'=' * 70}")
        log(rank, f"Done! Output: {output_path}")
        log(rank, f"{'=' * 70}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
