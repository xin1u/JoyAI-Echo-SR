"""Raw audio-video dataset loader for AV restoration training.

Loads MP4 + caption JSON pairs from a flat directory:
    data_root/
    ├── {uuid}.mp4
    └── {uuid}_caption.json

Each caption JSON has:
    {"caption_en": {"Summary": "...", "Style": "...", ...}, "caption_zh": {...}}

Returns raw pixel video (C, F, H, W) in [-1, 1] and audio waveform (C, T) ready
for degradation and VAE encoding.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltx_av_sr_trainer import logger


_CAPTION_FIELDS_FOR_PROMPT = [
    "Summary",
    "Roles_and_Subjects",
    "Style",
    "Camera_Movement",
    "Background",
    "BGM",
    "Sound_Effects",
    "Action_and_Dialogue",
]


class RawAVDataset(Dataset):
    """Dataset loading raw MP4 + caption JSON for AV restoration.

    Data layout (flat directory):
        data_root/{uuid}.mp4
        data_root/{uuid}_caption.json

    Args:
        data_root: Path to directory containing mp4 + caption files.
        target_width: Output video width in pixels (must be divisible by 32).
        target_height: Output video height in pixels (must be divisible by 32).
        target_frames: Number of frames to sample (must satisfy frames % 8 == 1).
        audio_sample_rate: Target audio sample rate.
        caption_lang: Which caption language to use ("en" or "zh").
        caption_fields: Which caption fields to concatenate into the prompt.
            None uses all default fields. Pass a list to select specific ones.
        require_audio: If True, skip samples that fail audio loading.
        seed: Random seed for reproducible temporal sampling.
    """

    def __init__(
        self,
        data_root: str,
        target_width: int = 576,
        target_height: int = 576,
        target_frames: int = 121,
        audio_sample_rate: int = 44100,
        caption_lang: str = "en",
        caption_fields: list[str] | None = None,
        require_audio: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.target_width = target_width
        self.target_height = target_height
        self.target_frames = target_frames
        self.audio_sample_rate = audio_sample_rate
        self.caption_lang = caption_lang
        self.caption_fields = caption_fields or list(_CAPTION_FIELDS_FOR_PROMPT)
        self.require_audio = require_audio
        self._rng = random.Random(seed)

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        if target_width % 32 != 0 or target_height % 32 != 0:
            raise ValueError(f"Resolution must be divisible by 32, got {target_width}x{target_height}")
        if target_frames % 8 != 1:
            raise ValueError(f"target_frames must satisfy frames % 8 == 1, got {target_frames}")

        self.samples = self._discover_samples()
        logger.info(f"RawAVDataset: {len(self.samples)} samples from {self.data_root}")

    def _discover_samples(self) -> list[tuple[Path, Path]]:
        """Find all (mp4, caption_json) pairs."""
        import os
        all_files = set(os.listdir(self.data_root))
        mp4_files = sorted(f for f in all_files if f.endswith(".mp4") and not f.endswith("_caption.mp4"))
        samples = []
        for fname in mp4_files:
            stem = fname[:-4]  # strip .mp4
            caption_name = f"{stem}_caption.json"
            if caption_name in all_files:
                samples.append((self.data_root / fname, self.data_root / caption_name))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | float]:
        mp4_path, caption_path = self.samples[index]

        video, fps, clip_start, clip_duration = self._load_video(mp4_path)
        caption = self._load_caption(caption_path)
        audio, audio_sr = self._load_audio(mp4_path, clip_start, clip_duration)

        result: dict[str, Tensor | str | float] = {
            "video": video,
            "caption": caption,
            "fps": fps,
            "sample_id": mp4_path.stem,
        }
        if audio is not None:
            result["audio"] = audio
            result["audio_sample_rate"] = float(audio_sr)

        return result

    # Match official LTX-2.3 pipeline: 121 frames @ 25fps → duration = 121/25 = 4.84s
    CANONICAL_FPS = 25.0

    def _load_video(self, path: Path) -> tuple[Tensor, float, float, float]:
        """Load a clip and sample target_frames, matching official pipeline timing.

        Official LTX pipeline uses fps=24.0 with 121 frames, giving
        duration = 121/24 = 5.0417s. We extract a clip of exactly this
        duration so that fps, frame count, and audio latent token count
        all match what the model was pretrained on.

        Returns:
            video: (C, F, H, W) tensor in [-1, 1]
            fps: CANONICAL_FPS (24.0)
            clip_start: start time of the clip in seconds
            clip_duration: duration of the clip in seconds (= target_frames / 24.0)
        """
        import decord
        decord.bridge.set_bridge("torch")

        vr = decord.VideoReader(
            str(path),
            height=self.target_height,
            width=self.target_width,
        )
        source_fps = vr.get_avg_fps()
        total_frames = len(vr)
        total_duration = total_frames / source_fps

        clip_duration = float(self.target_frames) / self.CANONICAL_FPS
        clip_duration = min(clip_duration, total_duration)
        clip_frames = int(clip_duration * source_fps)

        if total_duration > clip_duration:
            max_start_sec = total_duration - clip_duration
            clip_start = self._rng.random() * max_start_sec
        else:
            clip_start = 0.0

        start_frame = int(clip_start * source_fps)
        end_frame = min(start_frame + clip_frames, total_frames)

        if end_frame - start_frame >= self.target_frames:
            indices = torch.linspace(start_frame, end_frame - 1, self.target_frames).long().tolist()
        else:
            indices = list(range(start_frame, end_frame))

        vframes = vr.get_batch(indices)  # (F, H, W, C) uint8 torch tensor
        vframes = vframes.permute(0, 3, 1, 2)  # (F, C, H, W)

        if vframes.shape[0] < self.target_frames:
            vframes = self._pad_frames_with_reverse(vframes, self.target_frames)

        video = vframes.float() / 255.0  # (F, C, H, W) [0, 1]
        video = video.permute(1, 0, 2, 3)  # (C, F, H, W)
        video = video * 2.0 - 1.0  # [-1, 1]
        return video, self.CANONICAL_FPS, clip_start, clip_duration

    @staticmethod
    def _pad_frames_with_reverse(frames: Tensor, target: int) -> Tensor:
        """Pad frames via forward-reverse looping until reaching target count."""
        segments = [frames]
        total = frames.shape[0]
        forward = False
        while total < target:
            seg = frames.flip(0) if not forward else frames
            need = target - total
            segments.append(seg[:need])
            total += min(seg.shape[0], need)
            forward = not forward
        return torch.cat(segments, dim=0)[:target]

    @staticmethod
    def _load_audio_pyav(path: Path) -> tuple[Tensor, int]:
        """Load audio from video file via PyAV, preserving original channels."""
        import av
        container = av.open(str(path), metadata_encoding="utf-8", metadata_errors="ignore")
        astream = container.streams.audio[0]
        sr = astream.rate
        layout = "mono" if astream.channels == 1 else "stereo"
        resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=sr)
        frames = []
        for frame in container.decode(audio=0):
            frame = resampler.resample(frame)
            for f in frame:
                arr = f.to_ndarray()  # (channels, samples) with s16p
                frames.append(torch.from_numpy(arr.copy()))
        container.close()
        if not frames:
            raise RuntimeError("No audio frames decoded")
        waveform = torch.cat(frames, dim=-1).float() / 32768.0  # (C, T) in [-1, 1]
        return waveform, sr

    def _load_audio(self, path: Path, clip_start: float, clip_duration: float) -> tuple[Tensor | None, int]:
        """Load audio for the clip window, trim/pad to match video duration.

        Matches official trainer: preserves original channels (stereo) and
        original sample rate. AudioProcessor handles resampling to 16kHz
        internally during VAE encoding.

        Returns:
            waveform: (C, T) tensor in [-1, 1], or None on failure
            sr: original sample rate
        """
        try:
            waveform, sr = self._load_audio_pyav(path)
        except Exception as e:
            if self.require_audio:
                logger.warning(f"Audio load failed for {path.name}: {e}")
            return None, self.audio_sample_rate

        # Keep original channels (stereo if available) — matches official trainer
        # If mono (1, T), keep as is; AudioProcessor handles both

        # Extract the clip window matching the video segment
        start_sample = int(clip_start * sr)
        target_samples = int(clip_duration * sr)
        waveform = waveform[..., start_sample:]

        if waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]
        elif waveform.shape[-1] < target_samples:
            padding = target_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, sr

    def _load_caption(self, path: Path) -> str:
        """Load caption JSON and build a prompt from structured fields."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        lang_key = f"caption_{self.caption_lang}"
        cap = data.get(lang_key)
        if cap is None:
            for fallback in ["caption_en", "caption_zh"]:
                cap = data.get(fallback)
                if cap is not None:
                    break

        if cap is None:
            return ""

        if isinstance(cap, str):
            return cap

        if isinstance(cap, dict):
            parts = []
            for field in self.caption_fields:
                val = cap.get(field)
                if val and isinstance(val, str):
                    parts.append(val)
            return " ".join(parts) if parts else str(cap)

        return str(cap)
