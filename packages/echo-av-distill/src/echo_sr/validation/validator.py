"""Echo SR Validator: every rank processes its own clip, saves independently.

Pattern: all ranks do FSDP forward together, each with its own clip.
Each rank saves its result to the shared output dir.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from echo_sr import logger


class EchoValidator:
    """Validation: each rank loads its own clip, all run denoise together, each saves output."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self._val_clips: list[Path] = []
        self._clip_idx = 0
        self._loaded_clip = None

    def _discover_clips(self) -> None:
        """Find all val clips and assign round-robin to ranks."""
        val_cfg = self.trainer.config.get("validation", {})
        clips_dir = val_cfg.get("val_clips_dir", "")
        if not clips_dir or not Path(clips_dir).is_dir():
            # Fallback: single clip
            single = val_cfg.get("val_video_path", "")
            if single and Path(single).exists():
                self._val_clips = [Path(single)]
            return
        self._val_clips = sorted(Path(clips_dir).glob("*.mp4"))

    def _get_my_clip(self) -> Path | None:
        """Get this rank's clip (round-robin assignment)."""
        if not self._val_clips:
            self._discover_clips()
        if not self._val_clips:
            return None
        rank = self.trainer.accelerator.process_index
        idx = (self._clip_idx + rank) % len(self._val_clips)
        return self._val_clips[idx]

    def run(self) -> None:
        """Run validation: alternating TRAIN (current batch) / VAL (fixed clips)."""
        self._toggle = getattr(self, "_toggle", 1)
        if self._toggle == 1:
            self._toggle = 0
            self._run_train_mode()
        else:
            self._toggle = 1
            self._run_val_mode()

    def _run_train_mode(self) -> None:
        """[TRAIN] Denoise current training batch — all ranks participate, each saves."""
        t = self.trainer
        batch = getattr(t, "_last_batch", None)
        if batch is None:
            logger.warning("[TRAIN] No batch available — skipping")
            return

        device = t.device
        rank = t.accelerator.process_index
        val_cfg = t.config.get("validation", {})
        inference_steps = val_cfg.get("inference_steps", 8)

        hq_latent = batch["hq_latents"]["latents"][0:1].to(device, torch.bfloat16)
        lq_latent = batch["lq_latents"]["latents"][0:1].to(device, torch.bfloat16)
        num_frames = batch["hq_latents"]["num_frames"][0].item()
        height = batch["hq_latents"]["height"][0].item()
        width = batch["hq_latents"]["width"][0].item()

        lq_audio_lat = batch.get("audio_lq_latents", {}).get("latents")
        if lq_audio_lat is not None:
            lq_audio_lat = lq_audio_lat[0:1].to(device, torch.bfloat16)

        logger.info(f"Validation [TRAIN] step {t.global_step}: rank {rank}, latent {list(hq_latent.shape)}")

        t0 = time.time()
        restored_latent, sr_audio_lat = self._denoise(
            lq_latent, lq_audio_lat, num_frames, height, width, inference_steps, device,
        )
        t_denoise = time.time() - t0

        # Save — every rank saves its own result
        t1 = time.time()
        output_dir = t.output_dir / "samples" / f"step_{t.global_step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        _td = getattr(t, "tiny_decoder", None)
        if _td is None:
            try:
                from echo_sr.validation.tiny_decoder import TAEHV
                val_cfg_td = t.config.get("validation", {})
                td_path = Path(val_cfg_td.get("tiny_decoder_path", ""))
                if td_path.exists():
                    _td = TAEHV(checkpoint_path=str(td_path)).to(t.device, torch.bfloat16).eval()
                    t.tiny_decoder = _td
            except Exception as e:
                logger.warning(f"TinyDecoder load failed: {e}")
        if _td is None:
            logger.info(f"  [TRAIN] rank {rank}: no decoder — skipping save")
            return

        fps = val_cfg.get("frame_rate", 25.0)
        with torch.no_grad():
            sr_video = _td.decode_video(restored_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16), parallel=True, show_progress_bar=False)
            lq_video = _td.decode_video(lq_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16), parallel=True, show_progress_bar=False)
            gt_video = _td.decode_video(hq_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16), parallel=True, show_progress_bar=False)

        # Audio: use raw waveforms from batch
        raw_audio = getattr(t, "_last_batch_raw_audio", {})
        lq_audio_wav = raw_audio.get("lq_audio")
        gt_audio_wav = raw_audio.get("hq_audio")
        audio_sr = raw_audio.get("audio_sr", 44100)
        if isinstance(audio_sr, torch.Tensor):
            audio_sr = int(audio_sr[0].item()) if audio_sr.dim() > 0 else int(audio_sr.item())
        if lq_audio_wav is not None:
            lq_audio_wav = lq_audio_wav[0]  # first sample in batch
        if gt_audio_wav is not None:
            gt_audio_wav = gt_audio_wav[0]

        # SR audio from denoise
        sr_audio_wav, sr_audio_sr = self._decode_audio_latents(sr_audio_lat)

        prefix = f"[TRAIN]_rank{rank:02d}"
        self._save_mp4(sr_video[0], sr_audio_wav, sr_audio_sr if sr_audio_wav is not None else audio_sr, output_dir / f"{prefix}_sr.mp4", fps)
        self._save_mp4(lq_video[0], lq_audio_wav, audio_sr, output_dir / f"{prefix}_lq.mp4", fps)
        self._save_mp4(gt_video[0], gt_audio_wav, audio_sr, output_dir / f"{prefix}_gt.mp4", fps)

        # Compare image
        mid = sr_video.shape[1] // 2
        from torchvision.utils import save_image
        sr_mid = sr_video[0, mid]
        lq_mid = lq_video[0, min(mid, lq_video.shape[1] - 1)]
        gt_mid = gt_video[0, mid]
        if lq_mid.shape[-2:] != sr_mid.shape[-2:]:
            lq_mid = F.interpolate(lq_mid.unsqueeze(0), size=sr_mid.shape[-2:], mode="bilinear").squeeze(0)
        save_image(torch.cat([lq_mid, sr_mid, gt_mid], dim=2), output_dir / f"{prefix}_compare.png")

        t_save = time.time() - t1
        logger.info(f"  [TRAIN] rank {rank}: denoise={t_denoise:.1f}s, save={t_save:.1f}s")

    def _run_val_mode(self) -> None:
        """[VAL] Each rank loads its own fixed clip, all denoise in parallel."""
        t = self.trainer
        device = t.device
        val_cfg = t.config.get("validation", {})
        inference_steps = val_cfg.get("inference_steps", 8)

        clip_path = self._get_my_clip()
        if clip_path is None:
            logger.warning("No val clips found — skipping validation")
            return

        rank = t.accelerator.process_index
        logger.info(f"Validation step {t.global_step}: rank {rank} → {clip_path.name}")

        t0 = time.time()

        # Load and encode video
        clip_data = self._load_clip(clip_path)
        lq_video = clip_data["lq_video"].unsqueeze(0).to(device, torch.bfloat16)

        from ltx_core.model.video_vae import TilingConfig
        with torch.no_grad():
            lq_latent = t.vae_encoder.tiled_encode(lq_video, TilingConfig.default())

        # Encode audio
        lq_audio_lat = None
        if t.audio_vae_encoder is not None and clip_data.get("audio") is not None:
            audio = clip_data["audio"].to(device, torch.bfloat16)
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
            if audio.shape[1] == 1:
                audio = audio.expand(-1, 2, -1)
            with torch.no_grad():
                from ltx_core.model.audio_vae import encode_audio
                from ltx_core.types import Audio
                lq_audio_lat = encode_audio(
                    Audio(waveform=audio, sampling_rate=clip_data["audio_sr"]),
                    t.audio_vae_encoder,
                )

        # Target dimensions
        data_cfg = t.config["data"]
        from ltx_core.types import SpatioTemporalScaleFactors
        scale = SpatioTemporalScaleFactors.default()
        hq_h = data_cfg["hq_resolution"][1] // scale.height
        hq_w = data_cfg["hq_resolution"][0] // scale.width
        num_frames = lq_latent.shape[2]
        t_encode = time.time() - t0

        # Denoise (all ranks participate in FSDP forward)
        t1 = time.time()
        restored_latent, sr_audio_lat = self._denoise(
            lq_latent, lq_audio_lat, num_frames, hq_h, hq_w, inference_steps, device,
        )
        t_denoise = time.time() - t1

        # Save results (every rank saves its own)
        t2 = time.time()
        self._save_results(
            restored_latent, lq_latent, lq_audio_lat, sr_audio_lat,
            clip_data, num_frames, hq_h, hq_w, rank,
        )
        t_save = time.time() - t2

        logger.info(f"  rank {rank}: encode={t_encode:.1f}s, denoise={t_denoise:.1f}s, save={t_save:.1f}s")

        # Advance clip index for next validation
        self._clip_idx += t.accelerator.num_processes
        torch.cuda.empty_cache()

    # ─── Clip loader ────────────────────────────────────────────────────

    def _load_clip(self, path: Path) -> dict:
        """Load a 121-frame clip: video + audio."""
        import av
        data_cfg = self.trainer.config["data"]
        lq_w, lq_h, _ = data_cfg["lq_resolution"]

        container = av.open(str(path))
        fps = float(container.streams.video[0].average_rate)

        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            if len(frames) >= 121:
                break
        container.close()

        video = torch.stack(frames)
        lq_video = F.interpolate(video, size=(lq_h, lq_w), mode="bilinear")
        lq_video = lq_video.permute(1, 0, 2, 3) * 2 - 1  # [3, T, H, W] [-1,1]

        # Audio
        audio_data = None
        audio_sr = 48000
        try:
            container = av.open(str(path))
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
                    target_samples = int(len(frames) / fps * audio_sr)
                    if audio_data.shape[-1] > target_samples:
                        audio_data = audio_data[..., :target_samples]
            container.close()
        except Exception:
            pass

        return {"lq_video": lq_video, "audio": audio_data, "audio_sr": audio_sr, "fps": fps}

    # ─── Denoise ────────────────────────────────────────────────────────

    def _denoise(
        self, lq_latent: torch.Tensor,
        lq_audio_lat: torch.Tensor | None,
        num_frames: int, height: int, width: int,
        inference_steps: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Joint video+audio Euler ODE denoising."""
        from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.tools import VideoLatentTools, AudioLatentTools
        from ltx_core.types import VideoLatentShape, VideoPixelShape, AudioLatentShape, SpatioTemporalScaleFactors
        from ltx_core.model.transformer.model import X0Model
        from ltx_core.components.noisers import GaussianNoiser

        t = self.trainer
        val_cfg = t.config.get("validation", {})
        frame_rate = val_cfg.get("frame_rate", 25.0)
        scale = SpatioTemporalScaleFactors.default()

        video_patchifier = VideoLatentPatchifier(patch_size=1)
        audio_patchifier = AudioPatchifier(patch_size=1)

        pixel_h = height * scale.height
        pixel_w = width * scale.width
        pixel_f = (num_frames - 1) * scale.time + 1
        pixel_shape = VideoPixelShape(batch=1, frames=pixel_f, height=pixel_h, width=pixel_w, fps=frame_rate)

        video_tools = VideoLatentTools(
            patchifier=video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
            fps=frame_rate, scale_factors=scale, causal_fix=True,
        )

        generator = torch.Generator(device=device).manual_seed(t.config.get("seed", 42))
        noiser = GaussianNoiser(generator)
        video_state = video_tools.create_initial_state(device, torch.bfloat16)
        video_state = noiser(video_state, noise_scale=1.0)

        # Audio
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

        # Video LQ condition (add small condition noise for inference, matching training distribution)
        if lq_latent.shape[-2:] != (height, width):
            lq_cond = lq_latent  # 5D for cross-res
        else:
            lq_cond = video_patchifier.patchify(lq_latent)

        val_cfg = t.config.get("validation", {})
        val_cond_noise = val_cfg.get("condition_noise", 0.1)
        if val_cond_noise > 0:
            if isinstance(lq_cond, torch.Tensor):
                lq_cond = lq_cond + torch.randn_like(lq_cond) * val_cond_noise
            if audio_cond is not None:
                audio_cond = audio_cond + torch.randn_like(audio_cond) * val_cond_noise

        # Sigma schedule
        scheduler = LTX2Scheduler()
        target_shape = video_tools.target_shape
        if inference_steps == 1:
            sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
        else:
            dummy_lat = torch.empty(1, 1, target_shape.frames, target_shape.height, target_shape.width, device=device)
            sigmas = scheduler.execute(steps=inference_steps, latent=dummy_lat).to(device).float()

        stepper = EulerDiffusionStep()
        v_pos = t.val_embeds.video_context_positive.to(device)
        a_pos = t.val_embeds.audio_context_positive.to(device) if t.val_embeds.audio_context_positive is not None else None
        x0_model = X0Model(t.transformer)

        aligned_lq = lq_cond if lq_cond.dim() == 5 else self._align_tokens(lq_cond, video_state.latent.shape[1], device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step_idx, sigma in enumerate(sigmas[:-1]):
                video_mod = Modality(
                    enabled=True, latent=video_state.latent,
                    sigma=sigma.repeat(video_state.latent.shape[0]),
                    timesteps=sigma * video_state.denoise_mask,
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

                video_state = replace(video_state, latent=stepper.step(
                    sample=video_mod.latent, denoised_sample=denoised_video,
                    sigmas=sigmas, step_index=step_idx,
                ))
                if audio_state is not None and denoised_audio is not None:
                    audio_state = replace(audio_state, latent=stepper.step(
                        sample=audio_mod.latent, denoised_sample=denoised_audio,
                        sigmas=sigmas, step_index=step_idx,
                    ))

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        restored_audio_lat = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            restored_audio_lat = audio_state.latent

        return video_state.latent, restored_audio_lat

    @staticmethod
    def _align_tokens(tokens, target_len, device):
        if tokens is None or tokens.dim() != 3:
            return tokens
        if tokens.shape[1] == target_len:
            return tokens
        if tokens.shape[1] > target_len:
            return tokens[:, :target_len]
        pad = torch.zeros(tokens.shape[0], target_len - tokens.shape[1], tokens.shape[2], device=device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=1)

    def _get_tiny_decoder(self):
        """Load or return cached TinyDecoder."""
        t = self.trainer
        _td = getattr(t, "tiny_decoder", None)
        if _td is None:
            try:
                from echo_sr.validation.tiny_decoder import TAEHV
                val_cfg = t.config.get("validation", {})
                td_path = Path(val_cfg.get("tiny_decoder_path", ""))
                if td_path.exists():
                    _td = TAEHV(checkpoint_path=str(td_path)).to(t.device, torch.bfloat16).eval()
                    t.tiny_decoder = _td
                    logger.info(f"TinyDecoder loaded: {td_path}")
                else:
                    logger.warning(f"TinyDecoder path not found: {td_path}")
            except Exception as e:
                logger.warning(f"TinyDecoder load failed: {e}")
        return _td

    # ─── Save results ───────────────────────────────────────────────────

    def _save_results(
        self,
        restored_latent: torch.Tensor,
        lq_latent: torch.Tensor,
        lq_audio_lat: torch.Tensor | None,
        sr_audio_lat: torch.Tensor | None,
        clip_data: dict,
        num_frames: int, height: int, width: int,
        rank: int,
    ) -> None:
        t = self.trainer
        output_dir = t.output_dir / "samples" / f"step_{t.global_step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # TinyDecoder
        _td = getattr(t, "tiny_decoder", None)
        if _td is None:
            try:
                from echo_sr.validation.tiny_decoder import TAEHV
                val_cfg = t.config.get("validation", {})
                td_path = Path(val_cfg.get("tiny_decoder_path", ""))
                if td_path.exists():
                    _td = TAEHV(checkpoint_path=str(td_path)).to(t.device, torch.bfloat16).eval()
                    t.tiny_decoder = _td
            except Exception as e:
                logger.warning(f"TinyDecoder load failed: {e}")

        if _td is None:
            logger.info(f"  rank {rank}: no decoder — skipping save")
            return

        fps = clip_data.get("fps", 25.0)

        with torch.no_grad():
            sr_video = _td.decode_video(
                restored_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                parallel=True, show_progress_bar=False,
            )
            lq_video = _td.decode_video(
                lq_latent.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                parallel=True, show_progress_bar=False,
            )

        # Decode SR audio
        sr_audio_wav, audio_sr = self._decode_audio_latents(sr_audio_lat)
        lq_audio_wav = clip_data.get("audio")
        if lq_audio_wav is not None and sr_audio_wav is not None:
            # Resample LQ audio to match SR sample rate
            import torchaudio
            if clip_data["audio_sr"] != audio_sr:
                lq_audio_wav = torchaudio.functional.resample(lq_audio_wav, clip_data["audio_sr"], audio_sr)

        # Save
        prefix = f"rank{rank:02d}"
        self._save_mp4(sr_video[0], sr_audio_wav, audio_sr, output_dir / f"{prefix}_sr.mp4", fps)
        self._save_mp4(lq_video[0], lq_audio_wav, audio_sr if sr_audio_wav is not None else clip_data["audio_sr"],
                       output_dir / f"{prefix}_lq.mp4", fps)

        # Mid-frame compare image
        mid = sr_video.shape[1] // 2
        from torchvision.utils import save_image
        sr_mid = sr_video[0, mid]
        lq_mid = lq_video[0, min(mid, lq_video.shape[1] - 1)]
        if lq_mid.shape[-2:] != sr_mid.shape[-2:]:
            lq_mid = F.interpolate(lq_mid.unsqueeze(0), size=sr_mid.shape[-2:], mode="bilinear").squeeze(0)
        save_image(torch.cat([lq_mid, sr_mid], dim=2), output_dir / f"{prefix}_compare.png")

    def _decode_audio_latents(self, audio_lat: torch.Tensor | None) -> tuple[torch.Tensor | None, int]:
        if audio_lat is None:
            return None, 48000
        t = self.trainer
        components = getattr(t, "components", None)
        if components is None:
            return None, 48000
        audio_decoder = getattr(components, "audio_vae_decoder", None)
        vocoder = getattr(components, "vocoder", None)
        if audio_decoder is None or vocoder is None:
            return None, 48000

        device = audio_lat.device
        try:
            audio_decoder = audio_decoder.to(device).eval()
            vocoder = vocoder.to(device).eval()
            with torch.no_grad():
                mel = audio_decoder(audio_lat)
                wav = vocoder(mel)
            audio_sr = 48000
            if hasattr(vocoder, "vocoder"):
                audio_sr = vocoder.vocoder.output_sampling_rate * 3
            result = wav[0].float().cpu()
            audio_decoder.cpu()
            vocoder.cpu()
            return result, audio_sr
        except Exception as e:
            logger.warning(f"Audio decode failed: {e}")
            audio_decoder.cpu()
            vocoder.cpu()
            return None, 48000

    # ─── MP4 writer ─────────────────────────────────────────────────────

    @staticmethod
    def _save_mp4(frames: torch.Tensor, audio: torch.Tensor | None, audio_sr: int, path: Path, fps: float) -> None:
        """Save [T, 3, H, W] float [0,1] video + optional audio as mp4."""
        import torchvision.io as tvio

        video_uint8 = (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu()

        if audio is None:
            try:
                tvio.write_video(str(path), video_uint8, fps=fps, video_codec="libx264")
            except Exception as e:
                logger.warning(f"Video save failed ({path.name}): {e}")
            return

        # Write video + audio via temp files + ffmpeg merge (same filesystem)
        with tempfile.TemporaryDirectory(dir=str(path.parent)) as tmpdir:
            tmp_video = Path(tmpdir) / "video.mp4"
            tmp_audio = Path(tmpdir) / "audio.wav"

            try:
                tvio.write_video(str(tmp_video), video_uint8, fps=fps, video_codec="libx264")
            except Exception as e:
                logger.warning(f"Video save failed ({path.name}): {e}")
                return

            try:
                import torchaudio
                audio_cpu = audio.float().cpu()
                if audio_cpu.dim() == 1:
                    audio_cpu = audio_cpu.unsqueeze(0)
                torchaudio.save(str(tmp_audio), audio_cpu, audio_sr)
            except Exception as e:
                logger.warning(f"Audio save failed ({path.name}): {e}, saving video only")
                shutil.move(str(tmp_video), str(path))
                return

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(tmp_audio),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(path)],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                shutil.move(str(tmp_video), str(path))
