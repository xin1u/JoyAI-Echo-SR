#!/usr/bin/env python3
"""Long-video SR inference with sliding window and multi-GPU parallelism.

Each GPU loads the full 22B transformer (bf16 ~44GB) and processes a subset of
windows.  Overlap regions are linearly blended on rank 0 after all ranks finish.

v3: STG + global noise + full audio enhancement via DIT (matching train.py validation).

Usage (8 GPU):
  torchrun --nproc_per_node=8 scripts/infer_sr_long.py

Usage (single GPU):
  python scripts/infer_sr_long.py
"""

from __future__ import annotations

import json
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
# Config — defaults below are repo-relative; override via CLI args (see
# parse_args at the bottom of this file).
# ---------------------------------------------------------------------------
MODEL_PATH = "checkpoints/ltx-2.3-22b-dev.safetensors"
TEXT_ENCODER_PATH = "checkpoints/gemma-3-12b"
CHECKPOINT_PATH = "checkpoints/echo-sr/av-sr-1k-multistep-step09900.safetensors"

INPUT_VIDEO = ""  # required CLI arg
PROMPT_FILE = ""  # optional JSON with per-shot "Summary" texts
DEFAULT_PROMPT = "A high quality video with detailed visuals and natural motion"

LQ_W, LQ_H = 1280, 736
HQ_W, HQ_H = 1920, 1152
WINDOW_FRAMES = 121
WINDOW_STRIDE = 97  # only feeds the crossfade ramp width; see compute_shot_windows
                    # for the real schedule (1-frame overlap inside a shot)
FPS_MODEL = 24.0
FRAMES_PER_SHOT = 241  # Each shot is exactly 241 frames

INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.0
STG_SCALE = 1.0
STG_BLOCKS = [28]
SEED = 42
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0")) or None  # 0 = all

OUTPUT_DIR = _PKG_ROOT / "outputs" / "sr_long_inference"

NEG_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out "
    "colors, excessive noise, grainy texture, poor lighting, flickering, motion "
    "blur, distorted proportions, artifacts, low quality"
)

AUDIO_SAMPLE_RATE = 44100  # audio loaded at this rate for LQ condition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def read_video_window(path: str, start_frame: int, num_frames: int) -> torch.Tensor:
    """Read a specific range of frames from video file.

    Returns:
        video: [C, F, H, W] float32 in [-1, 1]
    """
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

    video = torch.stack(frames)  # [F, H, W, 3]
    video = video.permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return video


def read_audio_window(path: str, start_sec: float, duration_sec: float, target_sr: int = AUDIO_SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    """Read a specific time range of audio from video file.

    Returns:
        audio: [C, T] float32 waveform
        sr: sample rate
    """
    container = av.open(path)
    if len(container.streams.audio) == 0:
        container.close()
        return torch.zeros(2, 0), target_sr

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

    # Slice to requested time range
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    audio = audio[:, start_sample:end_sample]

    return audio, sr


def read_full_audio(path: str) -> tuple[torch.Tensor, int]:
    """Read full audio track from video file."""
    container = av.open(path)
    if len(container.streams.audio) == 0:
        container.close()
        return torch.zeros(2, 0), 48000

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
    return audio, sr


def get_video_info(path: str) -> tuple[int, int, int]:
    """Get (total_frames, fps, audio_sr) without reading all data."""
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


def load_shot_prompts(prompt_file: str) -> list[str]:
    """Load per-shot Summary texts."""
    with open(prompt_file) as f:
        data = json.load(f)
    shots = data.get("shots", [])
    fallback = "A high quality video with detailed visuals and natural motion"
    return [s.get("Summary", "") or fallback for s in shots]


def compute_shot_windows(num_shots: int) -> list[tuple[int, int, int]]:
    """Compute (shot_idx, start_frame, end_frame) with 2 windows per shot.

    Each shot = 241 frames. 2 windows of 121 frames with 1-frame overlap:
      Window 1: [shot_start, shot_start + 121]
      Window 2: [shot_start + 120, shot_start + 241]
    """
    windows = []
    for shot_idx in range(num_shots):
        shot_start = shot_idx * FRAMES_PER_SHOT
        windows.append((shot_idx, shot_start, shot_start + 121))
        windows.append((shot_idx, shot_start + 120, shot_start + 241))
    return windows


def load_models(device: torch.device, rank: int):
    """Load all models for inference. Returns a dict of components."""
    from ltx_trainer.model_loader import (
        load_audio_vae_decoder,
        load_audio_vae_encoder,
        load_embeddings_processor,
        load_text_encoder,
        load_transformer,
        load_video_vae_decoder,
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

    # LoRA
    log(rank, "  Applying LoRA...")
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

    log(rank, "  Merging LoRA...")
    transformer = transformer.merge_and_unload()
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()
    log(rank, f"  GPU mem after transformer: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # Text encoder
    log(rank, "  Loading text encoder...")
    text_encoder = load_text_encoder(TEXT_ENCODER_PATH, device=device, dtype=torch.bfloat16)
    embeddings_processor = load_embeddings_processor(MODEL_PATH, device=device, dtype=torch.bfloat16)

    # VAE
    log(rank, "  Loading video VAE...")
    vae_encoder = load_video_vae_encoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    vae_encoder.eval()
    vae_decoder = load_video_vae_decoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    vae_decoder.eval()

    # Audio components
    log(rank, "  Loading audio VAE + vocoder...")
    audio_encoder = load_audio_vae_encoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    audio_encoder.eval()
    audio_decoder = load_audio_vae_decoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    audio_decoder.eval()
    vocoder = load_vocoder(MODEL_PATH, device="cpu", dtype=torch.bfloat16)
    vocoder.eval()

    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "embeddings_processor": embeddings_processor,
        "vae_encoder": vae_encoder,
        "vae_decoder": vae_decoder,
        "audio_encoder": audio_encoder,
        "audio_decoder": audio_decoder,
        "vocoder": vocoder,
    }


def encode_text(models: dict, prompt: str, neg_prompt: str, device: torch.device):
    """Encode text prompts into embeddings. Returns (v_pos, a_pos, v_neg, a_neg)."""
    te = models["text_encoder"]
    ep = models["embeddings_processor"]

    with torch.no_grad():
        pos_out = ep.process_hidden_states(
            *te.encode(prompt, padding_side="left"), padding_side="left"
        )
        neg_out = ep.process_hidden_states(
            *te.encode(neg_prompt, padding_side="left"), padding_side="left"
        )

    v_pos = pos_out.video_encoding.to(device)
    a_pos = pos_out.audio_encoding.to(device) if pos_out.audio_encoding is not None else None
    v_neg = neg_out.video_encoding.to(device)
    a_neg = neg_out.audio_encoding.to(device) if neg_out.audio_encoding is not None else None

    return v_pos, a_pos, v_neg, a_neg


def free_text_encoder(models: dict):
    """Free text encoder after all prompts are encoded."""
    if "text_encoder" in models:
        del models["text_encoder"]
    torch.cuda.empty_cache()


def build_sigma_schedule(num_frames: int, device: torch.device) -> torch.Tensor:
    """Build sigma schedule with shift cap for the given frame count."""
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


def generate_global_noise(
    total_frames: int, device: torch.device,
) -> torch.Tensor:
    """Generate a global noise field for the entire video in latent space.

    Returns raw latent noise [1, 128, T_latent, H_latent, W_latent] for consistent
    per-window slicing (overlap regions get identical noise).
    """
    from ltx_core.types import SpatioTemporalScaleFactors

    sf = SpatioTemporalScaleFactors.default()
    t_latent = (total_frames - 1) // sf.time + 1
    h_latent = HQ_H // sf.height
    w_latent = HQ_W // sf.width

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn(1, 128, t_latent, h_latent, w_latent, generator=generator)
    return noise


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process a single window: encode LQ -> denoise (video+audio) -> decode.

    Returns:
        hq_video: [C, F, H_hq, W_hq] float in [0, 1]
        hq_audio: [C, T] float waveform or None
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )
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

    # --- Audio: waveform -> latent -> patchify ---
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
            target_shape=AudioLatentShape.from_duration(
                batch=1, duration=window_duration,
            ),
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
    lq_video_cond = lq_latent  # 5D for CondSRPatchifyProj

    video_mod = Modality(
        enabled=True,
        latent=video_state.latent,
        sigma=sigmas[0].repeat(video_state.latent.shape[0]),
        timesteps=video_state.denoise_mask,
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
            timesteps=audio_state.denoise_mask,
            positions=audio_state.positions,
            context=a_pos,
            context_mask=None,
            cond_latent=aligned_lq_audio,
        )

    # --- STG perturbation config ---
    stg_ptb_config = None
    if STG_SCALE > 0 and STG_BLOCKS:
        perturbations = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=STG_BLOCKS),
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=STG_BLOCKS),
        ]
        stg_ptb_config = BatchedPerturbationConfig(
            perturbations=[PerturbationConfig(perturbations=perturbations)]
        )

    # --- Denoising loop (3 passes: positive + CFG + STG) ---
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

            # Pass 1: Positive
            denoised_video, denoised_audio = x0_model(
                video=video_mod, audio=audio_mod, perturbations=None
            )
            pos_video, pos_audio = denoised_video, denoised_audio

            # Pass 2: CFG negative
            if GUIDANCE_SCALE > 1.0:
                video_neg_mod = replace(video_mod, context=v_neg)
                audio_neg_mod = replace(audio_mod, context=a_neg) if audio_mod is not None else None
                neg_video, neg_audio = x0_model(
                    video=video_neg_mod, audio=audio_neg_mod, perturbations=None
                )
                denoised_video = denoised_video + (GUIDANCE_SCALE - 1.0) * (pos_video - neg_video)
                if audio_mod is not None and denoised_audio is not None and neg_audio is not None:
                    denoised_audio = denoised_audio + (GUIDANCE_SCALE - 1.0) * (pos_audio - neg_audio)

            # Pass 3: STG perturbation
            if stg_ptb_config is not None and STG_SCALE > 0:
                ptb_video, ptb_audio = x0_model(
                    video=video_mod, audio=audio_mod, perturbations=stg_ptb_config
                )
                denoised_video = denoised_video + STG_SCALE * (pos_video - ptb_video)
                if audio_mod is not None and denoised_audio is not None and ptb_audio is not None:
                    denoised_audio = denoised_audio + STG_SCALE * (pos_audio - ptb_audio)

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

    # --- Decode video ---
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)

    models["vae_decoder"].to(device)
    video_chunks = list(models["vae_decoder"].tiled_decode(video_state.latent, TilingConfig.default()))
    video_out = torch.cat(video_chunks, dim=2)
    models["vae_decoder"].to("cpu")
    torch.cuda.empty_cache()

    hq_video = video_out.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5

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

    return hq_video, hq_audio


def _make_crossfade_weights(n: int, ramp_len: int, is_first: bool, is_last: bool) -> torch.Tensor:
    """Create per-frame linear cross-fade weights [n].

    - is_first=True: no leading ramp (full weight from frame 0)
    - is_last=True:  no trailing ramp (full weight to last frame)
    - Otherwise: linear ramp up over first `ramp_len` frames,
                 linear ramp down over last `ramp_len` frames
    """
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
    """Blend overlapping video windows with linear cross-fade.

    Returns [C, F_total, H, W] in [0,1].
    """
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
    """Blend overlapping audio windows with linear cross-fade in waveform space.

    Args:
        results: List of (start_sec, end_sec, waveform [C, T])
        total_duration: Total duration in seconds
        sr: Output sample rate (from vocoder)
    Returns:
        Blended waveform [C, total_samples]
    """
    overlap_sec = float(WINDOW_FRAMES - WINDOW_STRIDE) / 25.0  # overlap in seconds
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


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Long-video AV SR inference (multi-step, sliding window)")
    p.add_argument("--input", required=True, help="Input LQ video (with audio track)")
    p.add_argument("--model-path", default=MODEL_PATH, help="Base model safetensors")
    p.add_argument("--text-encoder-path", default=TEXT_ENCODER_PATH, help="Gemma text encoder dir")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="AV SR LoRA checkpoint")
    p.add_argument("--prompt-file", default=None,
                   help="Optional JSON file with {'shots': [{'Summary': ...}, ...]} per-shot prompts")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fallback prompt when no prompt file is given")
    p.add_argument("--output-dir", default="outputs/sr_long_inference")
    p.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    p.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    p.add_argument("--hq-width", type=int, default=HQ_W,
                   help="Output width — 1920 for the 1K checkpoint, 2560 for the 2K checkpoint")
    p.add_argument("--hq-height", type=int, default=HQ_H,
                   help="Output height — 1152 for the 1K checkpoint, 1472 for the 2K checkpoint")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    global MODEL_PATH, TEXT_ENCODER_PATH, CHECKPOINT_PATH, INPUT_VIDEO, PROMPT_FILE
    global OUTPUT_DIR, INFERENCE_STEPS, GUIDANCE_SCALE, SEED, DEFAULT_PROMPT
    global HQ_W, HQ_H

    args = parse_args()
    MODEL_PATH = args.model_path
    TEXT_ENCODER_PATH = args.text_encoder_path
    CHECKPOINT_PATH = args.checkpoint
    INPUT_VIDEO = args.input
    PROMPT_FILE = args.prompt_file
    DEFAULT_PROMPT = args.prompt
    OUTPUT_DIR = Path(args.output_dir)
    INFERENCE_STEPS = args.steps
    GUIDANCE_SCALE = args.guidance_scale
    HQ_W = args.hq_width  # must match the checkpoint's CondSRPatchifyProj grid
    HQ_H = args.hq_height
    SEED = args.seed

    # Distributed setup
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
    log(rank, "Long Video SR Inference v3 (STG + Global Noise + Audio Enhancement)")
    log(rank, f"  Rank {rank}/{world_size}, device={device}")
    log(rank, f"  Checkpoint: {CHECKPOINT_PATH}")
    log(rank, "=" * 70)

    # -----------------------------------------------------------------------
    # 1. Read input metadata
    # -----------------------------------------------------------------------
    log(rank, "\n[1/6] Reading input video metadata...")
    total_frames, src_fps, audio_sr = get_video_info(INPUT_VIDEO)
    total_duration = total_frames / src_fps
    log(rank, f"  Video: {total_frames} frames, fps={src_fps}, duration={total_duration:.1f}s, audio_sr={audio_sr}")

    if PROMPT_FILE:
        shot_prompts = load_shot_prompts(PROMPT_FILE)
        num_shots = len(shot_prompts)
        log(rank, f"  Loaded {num_shots} per-shot prompts")
    else:
        num_shots = max(1, (total_frames + FRAMES_PER_SHOT - 1) // FRAMES_PER_SHOT)
        shot_prompts = [DEFAULT_PROMPT] * num_shots
        log(rank, f"  No prompt file — using fallback prompt for {num_shots} shots")

    # -----------------------------------------------------------------------
    # 2. Compute windows (2 per shot) + generate global noise
    # -----------------------------------------------------------------------
    windows = compute_shot_windows(num_shots)
    if MAX_WINDOWS:
        windows = windows[:MAX_WINDOWS]
    log(rank, f"\n[2/6] {len(windows)} windows (2 per shot, 1-frame overlap within shot, no cross-shot overlap)")

    log(rank, "  Generating global noise field (CPU)...")
    global_noise = generate_global_noise(total_frames, device)
    log(rank, f"  Global noise: {list(global_noise.shape)}")

    my_windows = [(i, w) for i, w in enumerate(windows) if i % world_size == rank]
    print(f"  [Rank {rank}] Processing {len(my_windows)} windows", flush=True)

    # -----------------------------------------------------------------------
    # 3. Load models
    # -----------------------------------------------------------------------
    log(rank, f"\n[3/6] Loading models (rank {rank})...")
    models = load_models(device, rank)

    # -----------------------------------------------------------------------
    # 4. Encode per-shot text prompts
    # -----------------------------------------------------------------------
    log(rank, "\n[4/6] Encoding per-shot text prompts...")
    needed_shots: set[int] = {shot_idx for _, (shot_idx, _, _) in my_windows}
    prompt_cache: dict[int, tuple] = {}
    for shot_idx in sorted(needed_shots):
        v_pos, a_pos, v_neg, a_neg = encode_text(models, shot_prompts[shot_idx], NEG_PROMPT, device)
        prompt_cache[shot_idx] = (v_pos, a_pos, v_neg, a_neg)
        log(rank, f"  Shot {shot_idx}: {shot_prompts[shot_idx][:80]}...")
    free_text_encoder(models)
    log(rank, f"  Encoded {len(prompt_cache)} unique shot prompts")

    # -----------------------------------------------------------------------
    # 5. Process windows (video + audio)
    # -----------------------------------------------------------------------
    log(rank, f"\n[5/6] Processing windows ({INFERENCE_STEPS} steps, CFG={GUIDANCE_SCALE}, STG={STG_SCALE})...")

    for wi, (win_idx, (shot_idx, start, end)) in enumerate(my_windows):
        n_frames = end - start
        start_sec = start / src_fps
        duration_sec = n_frames / src_fps
        v_pos, a_pos, v_neg, a_neg = prompt_cache[shot_idx]
        print(
            f"  [Rank {rank}] Window {wi+1}/{len(my_windows)} "
            f"(global {win_idx+1}/{len(windows)}): shot={shot_idx} frames [{start}:{end}] ({n_frames}f), "
            f"audio [{start_sec:.2f}s:{start_sec+duration_sec:.2f}s]",
            flush=True,
        )

        video_chunk = read_video_window(INPUT_VIDEO, start, n_frames)
        audio_chunk, chunk_audio_sr = read_audio_window(INPUT_VIDEO, start_sec, duration_sec)

        hq_video, hq_audio = process_window(
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
        )

        # Save results
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

    # -----------------------------------------------------------------------
    # 6. Gather + blend + save
    # -----------------------------------------------------------------------
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

        # Blend audio
        final_audio = None
        audio_out_sr = None
        if all_audio_results:
            audio_out_sr = int(getattr(models["vocoder"], "output_sampling_rate", 48000))

            blend_duration = blend_frames / src_fps
            final_audio = blend_audio_windows(all_audio_results, blend_duration, audio_out_sr)
            log(rank, f"  Blended audio: {list(final_audio.shape)}, sr={audio_out_sr}")

        from ltx_trainer.video_utils import save_video

        output_path = OUTPUT_DIR / "enhanced_v3.2.mp4"
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
