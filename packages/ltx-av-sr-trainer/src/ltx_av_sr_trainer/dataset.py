"""Dataset for AV restoration training.
Loads precomputed data: HQ latents, LQ latents, text conditions, and audio latents.
The preprocessing script handles VAE encoding and degradation offline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_av_sr_trainer import logger


class AVRestorationDataset(Dataset):
    """Dataset for audio-video restoration from precomputed latents.

    Expected directory layout:
        data_root/
        ├── hq_latents/       # HQ video latents  [C, F, H, W]
        ├── lq_latents/       # LQ video latents  [C, F, H, W]
        ├── conditions/       # Text embeddings
        ├── audio_hq_latents/ # HQ audio latents  [C, T, F]
        └── audio_lq_latents/ # LQ audio latents  [C, T, F]
    """

    REQUIRED_DIRS = ["hq_latents", "lq_latents", "conditions"]
    AUDIO_DIRS = ["audio_hq_latents", "audio_lq_latents"]

    def __init__(self, data_root: str, with_audio: bool = True) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.with_audio = with_audio

        self._validate_dirs()
        self.sample_ids = self._discover_samples()
        logger.info(f"AVRestorationDataset: {len(self.sample_ids)} samples from {self.data_root}")

    def _validate_dirs(self) -> None:
        dirs_needed = self.REQUIRED_DIRS + (self.AUDIO_DIRS if self.with_audio else [])
        for d in dirs_needed:
            p = self.data_root / d
            if not p.exists():
                raise FileNotFoundError(f"Required directory missing: {p}")

    def _discover_samples(self) -> list[str]:
        hq_dir = self.data_root / "hq_latents"
        ids = sorted(p.stem for p in hq_dir.glob("*.pt"))
        valid = []
        for sid in ids:
            ok = True
            for d in self.REQUIRED_DIRS:
                if not (self.data_root / d / f"{sid}.pt").exists():
                    ok = False
                    break
            if ok and self.with_audio:
                for d in self.AUDIO_DIRS:
                    if not (self.data_root / d / f"{sid}.pt").exists():
                        ok = False
                        break
            if ok:
                valid.append(sid)
        return valid

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, dict[str, Tensor]]:
        sid = self.sample_ids[index]

        hq_data = torch.load(self.data_root / "hq_latents" / f"{sid}.pt", map_location="cpu", weights_only=True)
        lq_data = torch.load(self.data_root / "lq_latents" / f"{sid}.pt", map_location="cpu", weights_only=True)
        cond_data = torch.load(self.data_root / "conditions" / f"{sid}.pt", map_location="cpu", weights_only=True)

        hq_data = self._normalize_latents(hq_data)
        lq_data = self._normalize_latents(lq_data)

        result = {
            "hq_latents": hq_data,
            "lq_latents": lq_data,
            "conditions": cond_data,
            "idx": index,
        }

        if self.with_audio:
            audio_hq = torch.load(
                self.data_root / "audio_hq_latents" / f"{sid}.pt", map_location="cpu", weights_only=True
            )
            audio_lq = torch.load(
                self.data_root / "audio_lq_latents" / f"{sid}.pt", map_location="cpu", weights_only=True
            )
            result["audio_hq_latents"] = audio_hq
            result["audio_lq_latents"] = audio_lq

        return result

    @staticmethod
    def _normalize_latents(data: dict) -> dict:
        latents = data["latents"]
        if latents.dim() == 2:
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]
            latents = rearrange(latents, "(f h w) c -> c f h w", f=num_frames, h=height, w=width)
            data = data.copy()
            data["latents"] = latents
        return data
