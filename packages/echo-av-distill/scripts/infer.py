#!/usr/bin/env python3
"""Echo SR Inference with sliding window for long videos.

Supports videos of any length via overlapping 121-frame windows.
Each window uses the last frame of the previous SR output as first-frame condition.

For multi-GPU long-video inference with drop-first-frame chaining use
scripts/infer_av_distill_long.sh instead; this entry is the single-process
variant.

Usage:
    python packages/echo-av-distill/scripts/infer.py \
        --input input_736p.mp4 \
        --checkpoint checkpoints/echo-sr/av-sr-1k-distill-video-step005100.safetensors \
        --output outputs/infer/clip_sr.mp4 \
        --steps 10

Modified for the portable JoyAI-Echo-SR release in 2026.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Echo SR Inference")
    p.add_argument("--input", type=str, required=True, help="Input video path")
    p.add_argument("--checkpoint", type=str, required=True, help="LoRA + cond_proj safetensors")
    p.add_argument("--fallback_checkpoint", type=str, default=None, help="Teacher/fallback checkpoint to fill missing LoRA keys (e.g. audio modules)")
    p.add_argument("--config", type=str, default="configs/av_sr_1k_distill_video.yaml", help="Training config")
    p.add_argument("--output", type=str, default=None, help="Output video path")
    p.add_argument("--steps", type=int, default=10, help="Denoising steps")
    p.add_argument("--condition_noise", type=float, default=None, help="Condition noise level (default: from config validation.condition_noise)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_frames", type=int, default=121, help="Frames per window (must satisfy frames%%8==1)")
    p.add_argument("--overlap_frames", type=int, default=1, help="Overlap frames between windows (for blending)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use_tiny_decoder", action="store_true", help="Force TinyDecoder for VAE decoding (faster, lower quality)")
    return p.parse_args()


def load_full_video(path: str, lq_h: int, lq_w: int):
    """Load entire video, resize to LQ, return frames + audio."""
    import av

    container = av.open(path)
    fps = float(container.streams.video[0].average_rate)

    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    container.close()

    video = torch.stack(frames)  # [T, 3, H, W]
    orig_h, orig_w = video.shape[2], video.shape[3]
    if orig_h != lq_h or orig_w != lq_w:
        video = F.interpolate(video, size=(lq_h, lq_w), mode="bilinear")

    # Audio
    audio_data = None
    audio_sr = 48000
    try:
        container = av.open(path)
        if container.streams.audio:
            astream = container.streams.audio[0]
            audio_sr = astream.rate
            resampler = av.audio.resampler.AudioResampler(format="s16p", layout="stereo", rate=audio_sr)
            audio_frames = []
            for frame in container.decode(audio=0):
                for f in resampler.resample(frame):
                    audio_frames.append(torch.from_numpy(f.to_ndarray().copy()))
            if audio_frames:
                audio_data = torch.cat(audio_frames, dim=-1).float() / 32768.0
        container.close()
    except Exception:
        pass

    return video, audio_data, audio_sr, fps


def save_video_with_audio(frames: torch.Tensor, audio, audio_sr: int, path: Path, fps: float):
    import torchvision.io as tvio
    path.parent.mkdir(parents=True, exist_ok=True)
    video_uint8 = (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu()
    if audio is None:
        tvio.write_video(str(path), video_uint8, fps=fps, video_codec="libx264")
        return
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as tmpdir:
        tmp_v = Path(tmpdir) / "v.mp4"
        tmp_a = Path(tmpdir) / "a.wav"
        tvio.write_video(str(tmp_v), video_uint8, fps=fps, video_codec="libx264")
        import torchaudio
        a_cpu = audio.float().cpu()
        if a_cpu.dim() == 1: a_cpu = a_cpu.unsqueeze(0)
        torchaudio.save(str(tmp_a), a_cpu, audio_sr)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_v), "-i", str(tmp_a),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(path)],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            shutil.move(str(tmp_v), str(path))


def _align_tokens(tokens, target_len, device):
    if tokens is None or tokens.dim() != 3: return tokens
    if tokens.shape[1] == target_len: return tokens
    if tokens.shape[1] > target_len: return tokens[:, :target_len]
    pad = torch.zeros(tokens.shape[0], target_len - tokens.shape[1], tokens.shape[2], device=device, dtype=tokens.dtype)
    return torch.cat([tokens, pad], dim=1)


@torch.no_grad()
def denoise_window(
    transformer, lq_latent, lq_audio_lat, first_frame_latent,
    hq_h_lat, hq_w_lat, num_frames,
    v_pos, a_pos, device, args,
):
    """Denoise a single window with optional first-frame conditioning."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.tools import VideoLatentTools, AudioLatentTools
    from ltx_core.types import VideoLatentShape, VideoPixelShape, AudioLatentShape, SpatioTemporalScaleFactors

    scale = SpatioTemporalScaleFactors.default()
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    pixel_h = hq_h_lat * scale.height
    pixel_w = hq_w_lat * scale.width
    pixel_f = (num_frames - 1) * scale.time + 1
    fps = 25.0
    pixel_shape = VideoPixelShape(batch=1, frames=pixel_f, height=pixel_h, width=pixel_w, fps=fps)

    video_tools = VideoLatentTools(
        patchifier=video_patchifier,
        target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
        fps=fps, scale_factors=scale, causal_fix=True,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator)
    video_state = video_tools.create_initial_state(device, torch.bfloat16)
    video_state = noiser(video_state, noise_scale=1.0)

    # First-frame conditioning: inject clean latent for first frame tokens
    if first_frame_latent is not None:
        first_frame_patchified = video_patchifier.patchify(first_frame_latent)
        n_first = first_frame_patchified.shape[1]  # h_lat * w_lat tokens
        # Replace first-frame tokens with clean latent
        new_latent = video_state.latent.clone()
        new_latent[:, :n_first] = first_frame_patchified.to(new_latent.dtype)
        # Set denoise_mask=0 for first frame (don't denoise these)
        new_mask = video_state.denoise_mask.clone()
        new_mask[:, :n_first] = 0.0
        video_state = replace(video_state, latent=new_latent, denoise_mask=new_mask)

    # Audio state
    audio_state = None
    audio_tools = None
    audio_cond = None
    if lq_audio_lat is not None:
        audio_tools = AudioLatentTools(
            patchifier=audio_patchifier,
            target_shape=AudioLatentShape.from_torch_shape(lq_audio_lat.shape),
        )
        audio_state = audio_tools.create_initial_state(device, torch.bfloat16)
        audio_state = noiser(audio_state, noise_scale=1.0)
        audio_cond = audio_patchifier.patchify(lq_audio_lat)
        if args.condition_noise > 0:
            audio_cond = audio_cond + torch.randn_like(audio_cond) * args.condition_noise

    # LQ video condition
    if lq_latent.shape[-2:] != (hq_h_lat, hq_w_lat):
        lq_cond = lq_latent
    else:
        lq_cond = video_patchifier.patchify(lq_latent)
    if args.condition_noise > 0 and isinstance(lq_cond, torch.Tensor):
        lq_cond = lq_cond + torch.randn_like(lq_cond) * args.condition_noise

    # Sigma schedule
    target_shape = video_tools.target_shape
    if args.steps == 1:
        # Distilled 1-step model: trained with FixedSigmaSampler(sigma=1.0)
        # Must start from pure noise (sigma=1.0), not scheduler's shifted value
        sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    else:
        scheduler = LTX2Scheduler()
        dummy_lat = torch.empty(1, 1, target_shape.frames, target_shape.height, target_shape.width, device=device)
        sigmas = scheduler.execute(steps=args.steps, latent=dummy_lat).to(device).float()

    stepper = EulerDiffusionStep()
    x0_model = X0Model(transformer)
    aligned_lq = lq_cond if lq_cond.dim() == 5 else _align_tokens(lq_cond, video_state.latent.shape[1], device)

    # Store clean state for conditioning mask (only needed with first-frame conditioning)
    clean_latent = video_state.latent.clone() if first_frame_latent is not None else None

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            # Per-token timesteps: conditioned tokens get 0, rest get sigma
            timesteps = sigma * video_state.denoise_mask

            video_mod = Modality(
                enabled=True, latent=video_state.latent,
                sigma=sigma.repeat(video_state.latent.shape[0]),
                timesteps=timesteps,
                positions=video_state.positions,
                context=v_pos, context_mask=None, cond_latent=aligned_lq,
            )

            audio_mod = None
            if audio_state is not None and a_pos is not None:
                audio_mod = Modality(
                    enabled=True, latent=audio_state.latent,
                    sigma=sigma.repeat(audio_state.latent.shape[0]),
                    timesteps=sigma * audio_state.denoise_mask,
                    positions=audio_state.positions,
                    context=a_pos, context_mask=None, cond_latent=audio_cond,
                )

            denoised_video, denoised_audio = x0_model(video=video_mod, audio=audio_mod, perturbations=None)

            # Preserve conditioned tokens (first frame) — only when mask has zeros
            if first_frame_latent is not None:
                denoised_video = denoised_video * video_state.denoise_mask + clean_latent.float() * (1 - video_state.denoise_mask)

            video_state = replace(video_state, latent=stepper.step(
                sample=video_mod.latent, denoised_sample=denoised_video,
                sigmas=sigmas, step_index=step_idx,
            ))
            if audio_state is not None and denoised_audio is not None:
                audio_state = replace(audio_state, latent=stepper.step(
                    sample=audio_mod.latent, denoised_sample=denoised_audio,
                    sigmas=sigmas, step_index=step_idx,
                ))

    # Unpatchify
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)

    restored_audio_lat = None
    if audio_state is not None and audio_tools is not None:
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        restored_audio_lat = audio_state.latent

    return video_state.latent, restored_audio_lat


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    hq_w, hq_h, _ = data_cfg["hq_resolution"]
    lq_w, lq_h, _ = data_cfg["lq_resolution"]

    if args.condition_noise is None:
        args.condition_noise = cfg.get("validation", {}).get("condition_noise", 0.3)

    print(f"Input:  {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: LQ {lq_w}x{lq_h} → HQ {hq_w}x{hq_h}")
    print(f"Steps: {args.steps}, window: {args.window_frames}f, overlap: {args.overlap_frames}f, cond_noise: {args.condition_noise}")

    # ─── Load model ───
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from echo_sr.model.loader import (
        load_transformer,
        load_vae_encoder,
        load_video_vae_decoder,
        load_audio_vae_encoder,
        init_cond_proj,
    )
    from echo_sr.model.lora import EchoLoRA

    print("Loading model...")
    t0 = time.time()
    transformer = load_transformer(model_cfg["model_path"], dtype=torch.bfloat16)

    from ltx_core.types import SpatioTemporalScaleFactors
    scale = SpatioTemporalScaleFactors.default()
    sr_spatial = None
    cond_proj_cfg = cfg.get("cond_proj")
    if cond_proj_cfg and cond_proj_cfg.get("type") == "latent_2x":
        pass
    elif lq_w != hq_w or lq_h != hq_h:
        sr_spatial = {
            "lq_h": lq_h // scale.height, "lq_w": lq_w // scale.width,
            "hq_h": hq_h // scale.height, "hq_w": hq_w // scale.width,
        }
    init_cond_proj(transformer, sr_spatial, cond_proj_cfg)

    lora_cfg = cfg["lora"]

    from safetensors.torch import load_file

    # Merge fallback checkpoint into main checkpoint BEFORE creating EchoLoRA
    # so that rank inference sees the correct shapes for missing modules
    merged_ckpt_path = args.checkpoint
    fallback_filled = 0
    if args.fallback_checkpoint:
        ckpt_sd = load_file(args.checkpoint)
        fallback_sd = load_file(args.fallback_checkpoint)
        for k, v in fallback_sd.items():
            if k not in ckpt_sd and ("lora_A" in k or "lora_B" in k):
                ckpt_sd[k] = v
                fallback_filled += 1
        if fallback_filled > 0:
            import tempfile
            merged_tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False, dir="/tmp")
            from safetensors.torch import save_file
            save_file(ckpt_sd, merged_tmp.name)
            merged_ckpt_path = merged_tmp.name
            print(f"Filled {fallback_filled // 2} missing LoRA modules from fallback: {Path(args.fallback_checkpoint).name}")
        del fallback_sd, ckpt_sd

    lora = EchoLoRA(
        transformer,
        target_modules=lora_cfg["target_modules"],
        rank=lora_cfg.get("rank", 384),
        alpha=lora_cfg.get("alpha", 384),
        checkpoint=merged_ckpt_path,
        lora_structure_from=model_cfg.get("lora_structure_from"),
        param_dtype=torch.bfloat16,
    )

    # Clean up temp file
    if merged_ckpt_path != args.checkpoint:
        Path(merged_ckpt_path).unlink(missing_ok=True)

    ckpt_sd = load_file(args.checkpoint)
    loaded_cond = 0
    for name, param in transformer.named_parameters():
        if "_cond_" in name and "proj" in name:
            key = f"diffusion_model.{name}"
            if key in ckpt_sd:
                param.data.copy_(ckpt_sd[key].to(dtype=param.dtype))
                loaded_cond += 1
    print(f"Loaded: {lora.num_parameters():,} LoRA + {loaded_cond} cond_proj in {time.time()-t0:.1f}s")

    transformer = transformer.to(device=device, dtype=torch.bfloat16).eval()

    # VAE
    vae_encoder = load_vae_encoder(model_cfg["model_path"], device)
    audio_vae_encoder = None
    if cfg["training_strategy"].get("with_audio", True):
        audio_vae_encoder = load_audio_vae_encoder(model_cfg["model_path"], device)

    # Prompt embeddings
    from echo_sr.training.prompts import SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT
    from echo_sr.model.loader import encode_fixed_prompts
    cache_path = Path(cfg.get("prompt_cache_path", "checkpoints/prompt/sr_prompt_embeddings.pt"))
    cond_feats, val_embeds = encode_fixed_prompts(
        model_cfg["model_path"], model_cfg["text_encoder_path"], device,
        SR_FIXED_PROMPT, DEFAULT_NEGATIVE_PROMPT, cache_path,
    )
    v_pos = val_embeds.video_context_positive.to(device)
    a_pos = val_embeds.audio_context_positive.to(device) if val_embeds.audio_context_positive is not None else None

    # Video VAE decoder (full quality, replaces TinyDecoder)
    vae_decoder = None
    if args.use_tiny_decoder:
        print("Using TinyDecoder (forced via --use_tiny_decoder)")
    else:
        try:
            vae_decoder = load_video_vae_decoder(model_cfg["model_path"], device, dtype=torch.bfloat16)
            print(f"Full VAE decoder loaded ({sum(p.numel() for p in vae_decoder.parameters()):,} params)")
        except Exception as e:
            print(f"WARNING: Failed to load full VAE decoder ({e}); falling back to TinyDecoder")
            vae_decoder = None
    if vae_decoder is None:
        from echo_sr.validation.tiny_decoder import TAEHV
        td_path = cfg.get("validation", {}).get("tiny_decoder_path", "checkpoints/tinydecoder/taeltx2_3_wide.pth")
        vae_decoder = TAEHV(checkpoint_path=td_path).to(device, torch.bfloat16).eval()
        print(f"TinyDecoder loaded ({td_path})")

    # ─── Load video ───
    print(f"Loading video: {args.input}")
    all_frames, audio_data, audio_sr, fps = load_full_video(args.input, lq_h, lq_w)
    total_frames = all_frames.shape[0]
    print(f"  {total_frames} frames @ {fps}fps = {total_frames/fps:.1f}s")

    # ─── Sliding window inference ───
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.audio_vae import encode_audio
    from ltx_core.types import Audio

    hq_h_lat = hq_h // scale.height
    hq_w_lat = hq_w // scale.width
    enc_tiling = TilingConfig.default()

    window = args.window_frames
    overlap = args.overlap_frames
    stride = window - overlap

    # Compute windows
    windows = []
    start = 0
    while start < total_frames:
        end = min(start + window, total_frames)
        # Ensure frame count satisfies (f-1) % 8 == 0
        n = end - start
        if n > 1:
            n = ((n - 1) // 8) * 8 + 1
            end = start + n
            if end > total_frames:
                end = total_frames
                n = end - start
                n = ((n - 1) // 8) * 8 + 1
                start = max(0, end - n)
                end = start + n
        windows.append((start, end))
        if end >= total_frames:
            break
        start += stride

    print(f"  {len(windows)} windows: {[(s,e) for s,e in windows]}")

    # Process each window
    all_sr_frames = []
    prev_last_frame_latent = None  # For first-frame conditioning

    for wi, (w_start, w_end) in enumerate(windows):
        n_frames = w_end - w_start
        n_latent_frames = (n_frames - 1) // 8 + 1

        print(f"\nWindow {wi+1}/{len(windows)}: frames [{w_start}:{w_end}] ({n_frames}f, {n_latent_frames} latent)")

        # Extract window frames
        window_frames = all_frames[w_start:w_end]  # [T, 3, H, W] in [0,1]
        lq_video = (window_frames.permute(1, 0, 2, 3) * 2 - 1).unsqueeze(0).to(device, torch.bfloat16)  # [1, 3, T, H, W]

        # Encode LQ video
        lq_latent = vae_encoder.tiled_encode(lq_video, enc_tiling)

        # Encode audio for this window
        lq_audio_lat = None
        if audio_vae_encoder is not None and audio_data is not None:
            audio_start = int(w_start / fps * audio_sr)
            audio_end = int(w_end / fps * audio_sr)
            window_audio = audio_data[:, audio_start:audio_end]
            if window_audio.shape[-1] > 0:
                a_in = window_audio.unsqueeze(0).to(device, torch.bfloat16)
                if a_in.shape[1] == 1:
                    a_in = a_in.expand(-1, 2, -1)
                lq_audio_lat = encode_audio(Audio(waveform=a_in, sampling_rate=audio_sr), audio_vae_encoder)

        # Denoise with first-frame conditioning (from previous window's last frame)
        t1 = time.time()
        restored_latent, sr_audio_lat = denoise_window(
            transformer, lq_latent, lq_audio_lat,
            first_frame_latent=prev_last_frame_latent,
            hq_h_lat=hq_h_lat, hq_w_lat=hq_w_lat, num_frames=n_latent_frames,
            v_pos=v_pos, a_pos=a_pos, device=device, args=args,
        )
        print(f"  Denoised in {time.time()-t1:.1f}s, shape={list(restored_latent.shape)}")

        # Save last frame latent for next window's first-frame condition
        # last latent frame: [1, 128, 1, H, W]
        prev_last_frame_latent = restored_latent[:, :, -1:, :, :]

        # Decode video with full VAE decoder
        from ltx_core.model.video_vae import TilingConfig as DecodeTilingConfig
        if hasattr(vae_decoder, 'tiled_decode'):
            # Full VAE decoder: returns Iterator[Tensor], each chunk [B, 3, t, H, W] in [-1,1]
            chunks = list(vae_decoder.tiled_decode(restored_latent, tiling_config=DecodeTilingConfig.default()))
            sr_pixels = torch.cat(chunks, dim=2)  # concat along time
            sr_video = ((sr_pixels + 1) / 2).clamp(0, 1)  # [-1,1] → [0,1]
            sr_video = sr_video.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W] → [B, T, 3, H, W]
        elif hasattr(vae_decoder, 'decode_video'):
            # TinyDecoder fallback: input NTCHW, output NTCHW [0,1]
            sr_ntchw = restored_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
            sr_video = vae_decoder.decode_video(sr_ntchw, parallel=True, show_progress_bar=False)
        else:
            raise RuntimeError("Unknown decoder type")

        # For non-first windows, skip the overlapping frames (already in previous output)
        if wi == 0:
            all_sr_frames.append(sr_video[0].cpu())
        else:
            # Skip overlap frames from this window
            skip = overlap
            all_sr_frames.append(sr_video[0, skip:].cpu())

        torch.cuda.empty_cache()

    # Concatenate all windows
    sr_full = torch.cat(all_sr_frames, dim=0)  # [T_total, 3, H, W]
    print(f"\nFull SR video: {list(sr_full.shape)}")

    # ─── Save ───
    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"outputs/infer/{stem}_sr.mp4"

    output_path = Path(args.output)
    print(f"Saving: {output_path}")
    save_video_with_audio(sr_full, audio_data, audio_sr, output_path, fps)

    print(f"\nDone!")
    print(f"  SR: {output_path} ({hq_w}x{hq_h}, {sr_full.shape[0]}f)")


if __name__ == "__main__":
    main()
