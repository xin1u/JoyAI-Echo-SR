from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _squeeze_leading_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 0 and x.shape[0] == 1:
        return x.squeeze(0)
    return x


class Stage2SRPairDataset(Dataset):
    """Dataset for pair-style stage-2 SR distillation `.pt` files.

    Expected sample keys include at least:
      - ``prompt``
      - ``video_trajectory``
      - ``stage_2_final_video_latent``
      - ``sigmas``
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        prompt_key: str = "prompt",
        video_trajectory_key: str = "video_trajectory",
        final_video_key: str = "stage_2_final_video_latent",
        sigma_key: str = "sigmas",
        fps_key: str | None = None,
        fps: float = 24.0,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Pair data root does not exist: {self.data_root}")

        self.prompt_key = prompt_key
        self.video_trajectory_key = video_trajectory_key
        self.final_video_key = final_video_key
        self.sigma_key = sigma_key
        self.fps_key = fps_key
        self.default_fps = fps

        # Only read top-level `.pt` files under `data_root`.
        # User explicitly requested to ignore any nested subdirectories.
        self.files = sorted(self.data_root.glob("*.pt"))
        if not self.files:
            raise ValueError(f"No top-level .pt files found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        sample = torch.load(path, map_location="cpu", weights_only=False)

        prompt = sample[self.prompt_key]
        video_trajectory = sample[self.video_trajectory_key]
        final_video = sample[self.final_video_key]
        sigmas = sample[self.sigma_key]

        video_trajectory = _squeeze_leading_batch(video_trajectory)
        final_video = _squeeze_leading_batch(final_video)
        sigmas = _squeeze_leading_batch(sigmas)

        if video_trajectory.dim() != 5:
            raise ValueError(
                f"Expected video trajectory to have shape [steps, C, F, H, W], got {tuple(video_trajectory.shape)}"
            )
        if final_video.dim() != 4:
            raise ValueError(f"Expected final video latent to have shape [C, F, H, W], got {tuple(final_video.shape)}")
        if sigmas.numel() < 1:
            raise ValueError("Sigma tensor is empty")

        video_in = video_trajectory[0].contiguous()
        sigma0 = sigmas.reshape(-1)[0].to(torch.float32)

        fps = self.default_fps
        if self.fps_key is not None and self.fps_key in sample:
            fps_value = sample[self.fps_key]
            if isinstance(fps_value, torch.Tensor):
                fps = float(fps_value.reshape(-1)[0].item())
            else:
                fps = float(fps_value)

        return {
            "prompt": prompt,
            "video_in": video_in,
            "video_target": final_video.contiguous(),
            "sigma0": sigma0,
            "fps": torch.tensor(fps, dtype=torch.float32),
            "path": str(path),
        }


def stage2_sr_pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [item["prompt"] for item in batch]
    paths = [item["path"] for item in batch]
    video_in = torch.stack([item["video_in"] for item in batch], dim=0)
    video_target = torch.stack([item["video_target"] for item in batch], dim=0)
    sigma0 = torch.stack([item["sigma0"] for item in batch], dim=0)
    fps = torch.stack([item["fps"] for item in batch], dim=0)

    return {
        "prompt": prompts,
        "video_in": video_in,
        "video_target": video_target,
        "sigma0": sigma0,
        "fps": fps,
        "path": paths,
    }
