"""Video SR dataset that reads webdataset tars via HuggingFace datasets + decord.

Replicates the data loading pipeline from UltraForcing's ImageVideoWebDataset
but simplified for fixed-resolution SR training (no bucket balancing).
"""

from __future__ import annotations

import gc
import io
import json
import logging
import math
import os
import random
from typing import Any, Callable

import datasets
import datasets.config
from datasets.packaged_modules.webdataset import webdataset as _wds_mod

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets.distributed import split_dataset_by_node
from einops import rearrange
from torch.utils.data import IterableDataset

decord.bridge.set_bridge("torch")
datasets.disable_caching()

# Disable HuggingFace's automatic video decoding (torchcodec) for .mp4 files.
# We want raw bytes so decord can decode with our chosen height/width.
_wds_mod.VIDEO_EXTENSIONS = []
_wds_mod.WebDataset.VIDEO_EXTENSIONS = []

logger = logging.getLogger(__name__)


def _get_from_dict(data: dict, path: str) -> Any:
    keys = path.split(".")
    current = data
    for key in keys:
        current = current[key]
    return current


def _resize_crop_normalize(
    x: torch.Tensor, target_size: tuple[int, int]
) -> torch.Tensor:
    h, w = x.shape[-2:]
    bh, bw = target_size
    scale = max(bh / h, bw / w)
    resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
    x = TF.resize(
        x, (resize_h, resize_w),
        interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
    )
    x = TF.center_crop(x, target_size)
    return x / 127.5 - 1.0


class VideoSRDataset(IterableDataset):
    """Simplified webdataset video reader for SR training.

    Reads .tar shards containing {mp4, json} pairs. Decodes videos with
    decord, crops/resizes to a fixed target resolution, and yields
    ``{"pixel": (C, T, H, W), "caption": str}`` samples in [-1, 1].
    """

    def __init__(
        self,
        video_data_files: list[str],
        target_height: int,
        target_width: int,
        target_frames: int,
        fps: int = 24,
        min_frames: int | None = None,
        caption_keys: list[str] | None = None,
        caption_sampling_prob: list[float] | None = None,
        seed: int = 42,
        shuffle: bool = True,
        buffer_size: int = 1000,
    ) -> None:
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.target_frames = target_frames
        self.min_frames = min_frames if min_frames is not None else target_frames
        self.fps = fps
        self.caption_keys = caption_keys or ["caption.mimo_sft.caption_en"]
        self.caption_sampling_prob = caption_sampling_prob or [1.0]
        self.seed = seed
        self.shuffle = shuffle
        self.buffer_size = buffer_size

        world_size, rank = self._get_distributed_info()
        self.rng = random.Random(seed + rank)

        resolved_files = self._resolve_data_files(video_data_files)

        self._dataset = datasets.load_dataset(
            "webdataset",
            data_files=resolved_files,
            streaming=True,
            split="train",
        )
        self._dataset = split_dataset_by_node(
            self._dataset, rank=rank, world_size=world_size,
        )
        if self.shuffle:
            self._dataset = self._dataset.shuffle(
                seed=self.seed + self._dataset.epoch,
                buffer_size=self.buffer_size,
            )
        self._iterator = iter(self._dataset)

        decode_short = min(target_height, target_width)
        self._video_decode_short_side = decode_short

    @staticmethod
    def _get_distributed_info() -> tuple[int, int]:
        return int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))

    @staticmethod
    def _resolve_data_files(data_files: list[str]) -> list[str]:
        resolved: list[str] = []
        for f in data_files:
            if f.lower().endswith(".json"):
                with open(f, "r") as fp:
                    content = json.load(fp)
                resolved.extend(content["resolved_files"])
            else:
                resolved.append(f)
        return resolved

    def _get_video_decode_size(
        self, height: int, width: int
    ) -> tuple[int, int]:
        short = min(height, width)
        target = self._video_decode_short_side
        if short <= target:
            return height, width
        scale = target / short
        return math.ceil(height * scale), math.ceil(width * scale)

    def _get_caption(self, json_data: dict) -> str:
        available: dict[str, str] = {}
        weights: list[float] = []
        for key, prob in zip(self.caption_keys, self.caption_sampling_prob):
            try:
                caption = str(_get_from_dict(json_data, key))
                available[key] = caption
                weights.append(prob)
            except Exception:
                continue
        if not available:
            raise ValueError(f"No valid captions found in json: {list(json_data.keys())}")
        normalized = np.array(weights) / np.sum(weights)
        chosen_key = self.rng.choices(
            list(available.keys()), weights=normalized.tolist(), k=1
        )[0]
        return available[chosen_key]

    def _process_video(
        self, video_bytes: bytes, json_data: dict
    ) -> tuple[torch.Tensor, str]:
        raw_h, raw_w = json_data["height"], json_data["width"]
        dec_h, dec_w = self._get_video_decode_size(raw_h, raw_w)
        vr = decord.VideoReader(
            io.BytesIO(video_bytes), height=dec_h, width=dec_w,
        )

        cur_fps = json_data.get("fps", vr.get_avg_fps())
        target_fps = json_data.get("fps_target", self.fps)
        interval = max(1, int(cur_fps // target_fps))

        max_n_frame = json_data.get("frame_num_set", 257)
        start_idx = json_data.get("high_quality_frame_index", 0)
        end_index = min(len(vr), start_idx + max_n_frame * interval)
        indices = list(range(start_idx, end_index, interval))

        fps_ratio = max(1, round(min(cur_fps, target_fps) / self.fps))
        indices = indices[::fps_ratio]

        indices = self._sample_frames(indices, self.target_frames)
        if len(indices) < self.target_frames:
            raise ValueError(
                f"Not enough frames: got {len(indices)}, need {self.target_frames}"
            )

        pixel = vr.get_batch(indices)
        pixel = rearrange(pixel, "t h w c -> t c h w")
        pixel = _resize_crop_normalize(
            pixel, (self.target_height, self.target_width)
        )
        pixel = rearrange(pixel, "t c h w -> c t h w")

        caption = self._get_caption(json_data)
        return pixel, caption

    def _mirror_pad_indices(self, indices: list[int], n: int) -> list[int]:
        """Extend *indices* to length *n* via mirror-reflect: [0,1,2] → [0,1,2,1,0,1,2,...]."""
        length = len(indices)
        if length == 0:
            return []
        if length == 1:
            return indices * n
        cycle = list(range(length)) + list(range(length - 2, 0, -1))
        return [indices[cycle[i % len(cycle)]] for i in range(n)]

    def _sample_frames(self, indices: list[int], n: int) -> list[int]:
        length = len(indices)
        if length >= n:
            if self.rng.random() < 0.4:
                idxs = np.linspace(0, length - 1, n).round().astype(int)
                return [indices[i] for i in idxs]
            else:
                max_start = length - n
                start = self.rng.randint(0, max_start)
                return indices[start: start + n]
        if length >= self.min_frames:
            return self._mirror_pad_indices(indices, n)
        return indices

    def __iter__(self):
        while True:
            try:
                item = next(self._iterator, None)
                if item is None:
                    self._dataset.set_epoch(self._dataset.epoch + 1)
                    self._iterator = iter(self._dataset)
                    item = next(self._iterator)
            except Exception as e:
                logger.error(f"Error reading from dataset: {e}")
                continue

            if "mp4" not in item:
                continue

            try:
                json_data = item["json"]
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                elif isinstance(json_data, bytes):
                    json_data = json.loads(json_data.decode("utf-8"))
                pixel, caption = self._process_video(item["mp4"], json_data)
            except Exception as e:
                logger.warning(f"Error processing video: {e}")
                continue

            yield {"pixel": pixel, "caption": caption}


def video_sr_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = torch.stack([item["pixel"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"pixel": pixels, "caption": captions}
