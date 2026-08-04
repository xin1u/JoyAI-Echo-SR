"""三步教师在线蒸馏 SR LoRA 训练脚本。

教师（frozen distilled LoRA）跑 3 步 Euler 去噪生成 GT，
学生（可训练 LoRA）在 sigma=0.909375 处做单步预测，以教师输出为目标蒸馏。

Modified for the Echo-SR release in 2026: portable configuration paths,
environment-based experiment credentials, and config-only validation.

用法:
    python scripts/train_stage2_sr_distill.py configs/ltx2_stage2_sr_distill_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer", "ltx-sr-trainer"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

try:
    import swanlab
except ImportError:
    swanlab = None

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from einops import rearrange
from safetensors.torch import save_file
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.blocks import VideoDecoder
from ltx_pipelines.utils.media_io import encode_video
from ltx_sr_trainer.datasets.video_sr_dataset import VideoSRDataset, video_sr_collate
from ltx_sr_trainer.degradation import apply_aigc_realbasicvsr_degradation, apply_realbasicvsr_degradation
from ltx_sr_trainer.native_lora import NativeLoRAManager
from ltx_sr_trainer.training_strategies.stage2_sr_online import Stage2SROnlineConfig, Stage2SROnlineStrategy
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
    load_model as load_ltx_model,
)
from ltx_trainer.training_strategies.base_strategy import DEFAULT_FPS

# ---------------------------------------------------------------------------
# 工具函数（配置解析、LoRA 检查点、文本编码、保存等）
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三步教师在线蒸馏 SR LoRA 训练")
    parser.add_argument("config", type=str, help="YAML 配置文件路径")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the YAML structure without loading model weights or CUDA.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def validate_config_structure(cfg: dict[str, Any]) -> None:
    required = {
        "model": ("model_path", "text_encoder_path", "init_lora_path", "spatial_upsampler_path"),
        "data": ("train_data_files", "target_height", "target_width", "target_frames"),
        "training": ("output_dir", "sigma0"),
        "optimization": ("learning_rate", "steps", "batch_size"),
    }
    for section, keys in required.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Missing or invalid config section: {section}")
        missing = [key for key in keys if values.get(key) in (None, "", [])]
        if missing:
            raise ValueError(f"Missing required keys in {section}: {', '.join(missing)}")


def _normalize_path_string(value: str | Path | None) -> str | Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    return value.strip().strip("\"'")


def normalize_config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    if "model" in cfg:
        cfg["model"] = dict(cfg["model"])
        for key in ("model_path", "text_encoder_path", "init_lora_path", "spatial_upsampler_path"):
            if key in cfg["model"]:
                cfg["model"][key] = _normalize_path_string(cfg["model"].get(key))
    if "data" in cfg:
        cfg["data"] = dict(cfg["data"])
    if "training" in cfg:
        cfg["training"] = dict(cfg["training"])
        if "output_dir" in cfg["training"]:
            cfg["training"]["output_dir"] = _normalize_path_string(cfg["training"].get("output_dir"))
    return cfg


def inspect_lora_checkpoint(path: str | Path) -> dict[str, Any]:
    """读取 LoRA checkpoint，推断 rank / alpha / target_modules。"""
    from safetensors import safe_open

    path = Path(_normalize_path_string(path)).expanduser().resolve()
    rank: int | None = None
    alpha: int | None = None
    module_ranks: dict[str, int] = {}
    unique_ranks: set[int] = set()
    pattern = re.compile(r"^diffusion_model\.(.+)\.lora_[AB]\.weight$")
    target_set: set[str] = set()

    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        if "lora_alpha" in metadata:
            try:
                alpha = int(metadata["lora_alpha"])
            except Exception:
                alpha = None
        for key in f.keys():
            match = pattern.match(key)
            if not match:
                continue
            module_name = match.group(1)
            target_set.add(module_name)
            if key.endswith(".lora_A.weight"):
                shape = f.get_tensor(key).shape
                if len(shape) == 2:
                    module_rank = int(shape[0])
                    module_ranks[module_name] = module_rank
                    unique_ranks.add(module_rank)
                    if rank is None:
                        rank = module_rank

    if rank is None:
        raise ValueError(f"无法从 checkpoint 推断 LoRA rank: {path}")
    if alpha is None:
        alpha = rank
    target_modules = sorted(target_set)
    if not target_modules:
        raise ValueError(f"无法从 checkpoint 推断 LoRA target modules: {path}")

    return {
        "rank": rank, "alpha": alpha,
        "target_modules": target_modules,
        "target_module_count": len(target_modules),
        "module_ranks": module_ranks,
        "module_alphas": {m: alpha for m in target_modules},
        "unique_ranks": sorted(unique_ranks),
        "path": str(path),
    }


def maybe_sync_lora_hparams_from_init(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    init_lora_path = cfg.get("model", {}).get("init_lora_path")
    if not init_lora_path:
        return cfg, None
    inferred = inspect_lora_checkpoint(init_lora_path)
    cfg = dict(cfg)
    cfg["lora"] = dict(cfg["lora"])
    cfg["lora"]["rank"] = inferred["rank"]
    cfg["lora"]["alpha"] = inferred["alpha"]
    cfg["lora"]["target_modules"] = inferred["target_modules"]
    return cfg, inferred


def build_lora_metadata(cfg: dict[str, Any], inferred: dict[str, Any] | None = None) -> dict[str, str]:
    metadata = {
        "stage2_sr_distill": "true",
        "stage2_noise_sigma": str(cfg["training"].get("sigma0", 0.909375)),
        "lora_rank": str(cfg["lora"].get("rank", 32)),
        "lora_alpha": str(cfg["lora"].get("alpha", 32)),
        "lora_target_modules": json.dumps(cfg["lora"]["target_modules"]),
    }
    if inferred is not None:
        metadata["init_lora_path"] = inferred["path"]
        metadata["lora_spec_source"] = "init_lora_checkpoint"
    return metadata


def encode_prompts(
    prompts: list[str], text_encoder: Any, embeddings_processor: Any,
) -> dict[str, Tensor | None]:
    video_embeddings, audio_embeddings, attention_masks = [], [], []
    with torch.no_grad():
        for prompt in prompts:
            hidden_states, attention_mask = text_encoder.encode(prompt)
            out = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
            video_embeddings.append(out.video_encoding)
            if out.audio_encoding is not None:
                audio_embeddings.append(out.audio_encoding)
            attention_masks.append(out.attention_mask)
    return {
        "video_prompt_embeds": torch.cat(video_embeddings, dim=0),
        "prompt_attention_mask": torch.cat(attention_masks, dim=0),
        "audio_prompt_embeds": torch.cat(audio_embeddings, dim=0) if audio_embeddings else None,
    }


def save_lora_checkpoint(
    accelerator: Accelerator, model: Any, lora_manager: NativeLoRAManager,
    output_dir: Path, step: int, metadata: dict[str, str],
) -> Path | None:
    accelerator.wait_for_everyone()
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    fsdp_state_dict = accelerator.get_state_dict(model) if is_fsdp else None
    if not accelerator.is_main_process:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {
        k: v.to(torch.bfloat16) if isinstance(v, Tensor) else v
        for k, v in lora_manager.export_state_dict(model_state_dict=fsdp_state_dict).items()
    }
    save_path = output_dir / f"lora_weights_step_{step:05d}.safetensors"
    save_file(lora_state, save_path, metadata=metadata)
    return save_path

# ---------------------------------------------------------------------------
# 退化 + VAE 编码 + 像素损失辅助函数
# ---------------------------------------------------------------------------

def build_low_quality_video(
    video: Tensor, mode: str, rng: random.Random, generator: torch.Generator | None = None,
) -> Tensor:
    if generator is None:
        generator = torch.Generator(device=video.device).manual_seed(rng.randint(0, 2**31 - 1))
    if mode == "aigc_realbasic":
        return apply_aigc_realbasicvsr_degradation(video, rng, generator)
    if mode in ("realbasicvsr", "realesrgan"):
        return apply_realbasicvsr_degradation(video, rng, generator)
    raise ValueError(f"未知退化模式: {mode}")


def downsample_condition_video(
    video: Tensor, scale_range: tuple[float, float], rng: random.Random,
) -> Tensor:
    """像素空间下采样。scale_range=(2.0, 2.0) 表示固定 2 倍下采样。"""
    factor = rng.uniform(*scale_range)
    scale = 1.0 / max(factor, 1e-8)
    if scale >= 0.999:
        return video
    b, c, t, h, w = video.shape
    target_h, target_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if target_h == h and target_w == w:
        return video
    video_2d = rearrange(video, "b c t h w -> (b t) c h w").contiguous().float()
    resized = F.interpolate(video_2d, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return rearrange(resized.to(dtype=video.dtype), "(b t) c h w -> b c t h w", b=b, t=t)


def compute_pixel_losses(
    vae_decoder: torch.nn.Module, lpips_fn: torch.nn.Module,
    pred_x0_latent: Tensor, gt_latent: Tensor, num_frames: int = 8,
) -> tuple[Tensor, Tensor]:
    """随机抽取 num_frames 帧 decode，计算像素空间 L1 + LPIPS 损失。"""
    Fp = pred_x0_latent.shape[2]
    if num_frames >= Fp:
        frame_indices = list(range(Fp))
    else:
        frame_indices = sorted(random.sample(range(Fp), num_frames))

    _dec_dtype = next(vae_decoder.parameters()).dtype
    l1_vals: list[Tensor] = []
    lpips_vals: list[Tensor] = []
    for idx in frame_indices:
        pred_slice = pred_x0_latent[:, :, idx : idx + 1, :, :]
        gt_slice = gt_latent[:, :, idx : idx + 1, :, :]
        pred_pixels = vae_decoder(pred_slice.to(_dec_dtype))
        with torch.no_grad():
            gt_pixels = vae_decoder(gt_slice.to(_dec_dtype))
        pred_frame = pred_pixels[:, :, 0]
        gt_frame = gt_pixels[:, :, 0].detach()
        l1_vals.append(F.l1_loss(pred_frame, gt_frame))
        pred_01 = (pred_frame + 1) * 0.5
        gt_01 = (gt_frame + 1) * 0.5
        lpips_vals.append(lpips_fn(pred_01.float(), gt_01.float()).mean())
    return torch.stack(l1_vals).mean(), torch.stack(lpips_vals).mean()

# ---------------------------------------------------------------------------
# 教师推理：3 步 Euler 去噪
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_euler_denoise(
    transformer: Any,
    teacher_lora: NativeLoRAManager,
    student_lora: NativeLoRAManager,
    strategy: Stage2SROnlineStrategy,
    *,
    lq_latent: Tensor,
    noise: Tensor,
    sigma_schedule: list[float],
    conditions: dict[str, Tensor | None],
    fps: float,
) -> Tensor:
    """教师 3 步 Euler 去噪：从 lq_latent + noise 出发，逐步去噪到 sigma=0。

    Args:
        transformer: 共享的 transformer 模型
        teacher_lora: 教师 LoRA 分支（frozen）
        student_lora: 学生 LoRA 分支（推理时关闭）
        strategy: 训练策略（提供 patchifier 和 position 计算）
        lq_latent: (B, C, F, H, W) 上采样后的低质量 latent
        noise: (B, C, F, H, W) 与 lq_latent 同形状的噪声
        sigma_schedule: [0.909375, 0.725, 0.421875, 0.0]
        conditions: 文本编码结果
        fps: 帧率

    Returns:
        teacher_x0: (B, C, F, H, W) 教师去噪后的干净 latent
    """
    # 切换到教师分支
    teacher_lora.set_enabled(True)
    student_lora.set_enabled(False)

    batch_size = lq_latent.shape[0]
    num_frames = lq_latent.shape[2]
    height = lq_latent.shape[3]
    width = lq_latent.shape[4]
    device = lq_latent.device
    dtype = lq_latent.dtype

    video_prompt_embeds = conditions["video_prompt_embeds"]
    prompt_attention_mask = conditions["prompt_attention_mask"]

    # 初始噪声状态：x_t = sigma_0 * noise + (1 - sigma_0) * lq_latent
    x_t = sigma_schedule[0] * noise + (1 - sigma_schedule[0]) * lq_latent

    # 3 步 Euler 去噪
    for i in range(len(sigma_schedule) - 1):
        sigma_cur = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]

        # patchify 当前状态
        video_tokens = strategy._video_patchifier.patchify(x_t)
        video_seq_len = video_tokens.shape[1]

        # 构建 Modality
        sigmas = torch.full((batch_size,), sigma_cur, device=device, dtype=dtype)
        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = strategy._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )
        video_modality = Modality(
            enabled=True, sigma=sigmas, latent=video_tokens,
            timesteps=video_timesteps, positions=video_positions,
            context=video_prompt_embeds, context_mask=prompt_attention_mask,
        )

        # 教师前向：预测 velocity
        velocity, _ = transformer(video=video_modality, audio=None, perturbations=None)

        # Euler 步进：x_{t-1} = x_t + (sigma_next - sigma_cur) * velocity
        # 注意：velocity 在 patchified token 空间，需要 unpatchify 回 latent 空间
        velocity_latent = strategy._video_patchifier.unpatchify(
            velocity,
            output_shape=VideoLatentShape(
                batch=batch_size, channels=128,
                frames=num_frames, height=height, width=width,
            ),
        )
        x_t = x_t + (sigma_next - sigma_cur) * velocity_latent

    # 恢复学生分支
    teacher_lora.set_enabled(False)
    student_lora.set_enabled(True)

    return x_t  # teacher_x0

# ---------------------------------------------------------------------------
# 验证辅助函数
# ---------------------------------------------------------------------------

def _materialize_video_iterator(video_iter: Any) -> Tensor:
    chunks = list(video_iter)
    if not chunks:
        raise RuntimeError("Video decoder returned no chunks")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def run_validation(
    *, accelerator: Accelerator, model: Any, strategy: Stage2SROnlineStrategy,
    student_lora: NativeLoRAManager, teacher_lora: NativeLoRAManager,
    video_decoder: VideoDecoder,
    teacher_x0: Tensor, lq_latent: Tensor, student_x0: Tensor,
    conditions: dict[str, Tensor | None],
    output_dir: Path, step: int, fps: float,
) -> None:
    """保存教师 GT / LQ / 学生 SR 对比视频。"""
    if not accelerator.is_main_process:
        return
    val_dir = output_dir / "validation" / f"step_{step:05d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    tiling_config = TilingConfig.default()
    decode_gen = torch.Generator(device=accelerator.device).manual_seed(42)

    def _decode(latent: Tensor) -> Tensor:
        return _materialize_video_iterator(
            video_decoder(latent[:1].to(torch.bfloat16), tiling_config, decode_gen)
        )

    sr_video = _decode(student_x0)
    teacher_video = _decode(teacher_x0)
    lq_video = _decode(lq_latent)

    fps_int = int(fps)
    encode_video(video=sr_video, fps=fps_int, audio=None, output_path=str(val_dir / "student_sr.mp4"), video_chunks_number=1)
    encode_video(video=teacher_video, fps=fps_int, audio=None, output_path=str(val_dir / "teacher_gt.mp4"), video_chunks_number=1)
    encode_video(video=lq_video, fps=fps_int, audio=None, output_path=str(val_dir / "lq.mp4"), video_chunks_number=1)

    min_f = min(sr_video.shape[0], teacher_video.shape[0], lq_video.shape[0])
    comparison = torch.cat([lq_video[:min_f], sr_video[:min_f], teacher_video[:min_f]], dim=2)
    encode_video(video=comparison, fps=fps_int, audio=None,
                 output_path=str(val_dir / "compare_lq_vs_student_vs_teacher.mp4"), video_chunks_number=1)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = normalize_config_paths(load_config(args.config))
    validate_config_structure(cfg)
    if args.validate_config:
        print(f"Config OK: {Path(args.config).resolve()}")
        return
    cfg, inferred_init_lora = maybe_sync_lora_hparams_from_init(cfg)

    # ---- Accelerator ----
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision_mode", "bf16"),
        gradient_accumulation_steps=cfg["optimization"].get("gradient_accumulation_steps", 1),
    )
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    if is_fsdp and str(cfg.get("lora", {}).get("param_dtype", "bf16")).lower() not in {"bf16", "bfloat16"}:
        cfg["lora"]["param_dtype"] = "bf16"
    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg["training"]["output_dir"]).expanduser().resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    accelerator.wait_for_everyone()

    # ---- SwanLab ----
    swanlab_cfg = cfg.get("swanlab", {})
    enable_swanlab = bool(swanlab_cfg.get("enable_swanlab", False)) and swanlab is not None
    if enable_swanlab and accelerator.is_main_process:
        swanlab_api_key = os.environ.get("SWANLAB_API_KEY")
        if swanlab_api_key:
            swanlab.login(api_key=str(swanlab_api_key))
        exp_name = str(swanlab_cfg.get("swanlab_experiment_name", "")) or Path(cfg["training"]["output_dir"]).name
        swanlab.init(
            project=str(swanlab_cfg.get("swanlab_project", "LTX-2.3-stage2-sr-distill")),
            experiment_name=exp_name, config=cfg,
        )

    # ---- 策略配置 ----
    deg_cfg = cfg.get("degradation", {})
    strategy_cfg = Stage2SROnlineConfig(
        sigma0=cfg["training"].get("sigma0", 0.909375),
        velocity_loss_weight=cfg["training"].get("velocity_loss_weight", 1.0),
        x0_loss_weight=cfg["training"].get("x0_loss_weight", 0.25),
        # 蒸馏模式不对 LQ 做条件增强
        condition_noise_min=0.0, condition_noise_max=0.0,
        condition_drop_prob_min=0.0, condition_drop_prob_max=0.0,
    )
    strategy = Stage2SROnlineStrategy(strategy_cfg)
    save_metadata = build_lora_metadata(cfg, inferred_init_lora)

    # ---- 教师 sigma schedule ----
    teacher_cfg = cfg.get("teacher", {})
    teacher_sigma_schedule = teacher_cfg.get("sigma_schedule", [0.909375, 0.725, 0.421875, 0.0])

    # ---- 数据集 ----
    data_cfg = cfg["data"]
    dataset = VideoSRDataset(
        video_data_files=data_cfg["train_data_files"],
        target_height=data_cfg["target_height"],
        target_width=data_cfg["target_width"],
        target_frames=data_cfg["target_frames"],
        fps=data_cfg.get("fps", 24),
        caption_keys=data_cfg.get("caption_keys"),
        caption_sampling_prob=data_cfg.get("caption_sampling_prob"),
        seed=cfg.get("seed", 42),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["optimization"].get("batch_size", 1),
        num_workers=data_cfg.get("num_dataloader_workers", 2),
        pin_memory=data_cfg.get("num_dataloader_workers", 2) > 0,
        persistent_workers=data_cfg.get("num_dataloader_workers", 2) > 0,
        collate_fn=video_sr_collate,
    )

    # ---- Transformer + 双 LoRA（教师 frozen + 学生 trainable） ----
    components = load_ltx_model(
        checkpoint_path=cfg["model"]["model_path"],
        device="cpu", dtype=torch.bfloat16,
        with_video_vae_encoder=False, with_video_vae_decoder=False,
        with_audio_vae_decoder=False, with_vocoder=False, with_text_encoder=False,
    )
    transformer = components.transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)

    param_dtype_str = cfg["lora"].get("param_dtype", "bf16")
    param_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[param_dtype_str]

    # 教师 LoRA：frozen，用于 3 步推理生成 GT
    teacher_lora = NativeLoRAManager(
        transformer,
        target_modules=cfg["lora"]["target_modules"],
        rank=cfg["lora"].get("rank", 32),
        alpha=cfg["lora"].get("alpha", 32),
        module_ranks=inferred_init_lora["module_ranks"] if inferred_init_lora else None,
        module_alphas=inferred_init_lora["module_alphas"] if inferred_init_lora else None,
        dropout=0.0,
        param_dtype=param_dtype,
        attach_to_targets=is_fsdp,
        namespace="teacher",
    )
    init_lora_path = cfg["model"].get("init_lora_path")
    if init_lora_path:
        teacher_lora.load_lora_weights(init_lora_path)
    teacher_lora.requires_grad_(False)
    teacher_lora.set_enabled(False)

    # 学生 LoRA：可训练，从教师 LoRA 初始化，单步蒸馏
    student_lora = NativeLoRAManager(
        transformer,
        target_modules=cfg["lora"]["target_modules"],
        rank=cfg["lora"].get("rank", 32),
        alpha=cfg["lora"].get("alpha", 32),
        module_ranks=inferred_init_lora["module_ranks"] if inferred_init_lora else None,
        module_alphas=inferred_init_lora["module_alphas"] if inferred_init_lora else None,
        dropout=cfg["lora"].get("dropout", 0.0),
        param_dtype=param_dtype,
        attach_to_targets=is_fsdp,
        namespace="student",
    )
    if init_lora_path:
        student_lora.load_lora_weights(init_lora_path)
    student_lora.set_enabled(True)

    # 冻结学生 audio 相关 LoRA 参数（SR 只训练 video 分支）
    audio_frozen_count = 0
    for name, param in student_lora.named_lora_parameters():
        if "audio" in name:
            param.requires_grad_(False)
            audio_frozen_count += 1

    transformer.set_gradient_checkpointing(cfg["optimization"].get("enable_gradient_checkpointing", True))
    transformer._teacher_lora = teacher_lora  # type: ignore[attr-defined]
    transformer._student_lora = student_lora  # type: ignore[attr-defined]
    transformer.train()

    # 优化器：只优化学生 LoRA 参数
    trainable_params = [p for p in student_lora.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg["optimization"].get("learning_rate", 5e-5),
        betas=tuple(cfg["optimization"].get("adam_betas", [0.9, 0.95])),
        weight_decay=cfg["optimization"].get("weight_decay", 0.0),
        eps=cfg["optimization"].get("adam_eps", 1e-8),
        foreach=cfg["optimization"].get("adam_foreach", False),
    )

    if accelerator.is_main_process:
        trainable_count = sum(p.numel() for p in trainable_params)
        total_student = sum(1 for _ in student_lora.named_lora_parameters())
        total_teacher = sum(1 for _ in teacher_lora.named_lora_parameters())
        print(f"# 学生可训练 LoRA 参数: {trainable_count:,}")
        print(f"# 学生 LoRA targets: {len(student_lora.resolved_target_modules)} "
              f"(audio frozen: {audio_frozen_count}/{total_student})")
        print(f"# 教师 LoRA targets: {len(teacher_lora.resolved_target_modules)} (全部 frozen)")
        print(f"# LoRA rank={cfg['lora']['rank']}, alpha={cfg['lora']['alpha']}")
        if init_lora_path:
            print(f"# 教师/学生 LoRA 初始化自: {init_lora_path}")

    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    # ---- VAE encoder（编码视频到 latent 空间） ----
    vae_encoder = load_video_vae_encoder(
        checkpoint_path=cfg["model"]["model_path"],
        device=accelerator.device, dtype=torch.bfloat16,
    )
    vae_encoder.requires_grad_(False)
    vae_encoder.eval()

    # ---- Spatial upsampler（latent 2x 上采样，匹配 stage2 pipeline） ----
    spatial_upsampler = None
    spatial_upsampler_path = cfg["model"].get("spatial_upsampler_path")
    if spatial_upsampler_path:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as _Builder
        spatial_upsampler = _Builder(
            model_path=str(spatial_upsampler_path),
            model_class_configurator=LatentUpsamplerConfigurator,
        ).build(device=accelerator.device, dtype=torch.bfloat16)
        spatial_upsampler.requires_grad_(False)
        spatial_upsampler.eval()
        if accelerator.is_main_process:
            print(f"# Spatial upsampler: {spatial_upsampler_path}")

    # ---- 文本编码器 ----
    text_encoder = load_text_encoder(
        gemma_model_path=cfg["model"]["text_encoder_path"],
        device=accelerator.device, dtype=torch.bfloat16,
        load_in_8bit=cfg["model"].get("load_text_encoder_in_8bit", False),
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=cfg["model"]["model_path"],
        device=accelerator.device, dtype=torch.bfloat16,
    )
    text_encoder.requires_grad_(False).eval()
    embeddings_processor.requires_grad_(False).eval()

    # ---- 像素损失：VAE decoder + LPIPS（可选） ----
    pixel_loss_cfg = cfg.get("pixel_losses", {})
    pixel_loss_enabled = bool(pixel_loss_cfg.get("enabled", False))
    vae_decoder_nn = None
    lpips_fn = None
    if pixel_loss_enabled:
        vae_decoder_nn = load_video_vae_decoder(
            cfg["model"]["model_path"], device=accelerator.device, dtype=torch.bfloat16,
        )
        vae_decoder_nn.requires_grad_(False)
        vae_decoder_nn.eval()

        import lpips as _lpips
        lpips_weights_path = pixel_loss_cfg.get("lpips_weights_path")
        if not lpips_weights_path:
            raise ValueError("pixel_losses.lpips_weights_path is required when pixel losses are enabled")
        lpips_fn = _lpips.LPIPS(net="vgg", pnet_rand=True, pretrained=False, verbose=False)
        lpips_fn.load_state_dict(torch.load(lpips_weights_path, map_location="cpu"))
        lpips_fn = lpips_fn.to(device=accelerator.device)
        lpips_fn.requires_grad_(False)
        lpips_fn.eval()

        if accelerator.is_main_process:
            dec_params = sum(p.numel() for p in vae_decoder_nn.parameters())
            print(f"# 像素损失已启用: VAE decoder {dec_params:,} params, LPIPS VGG loaded.")

    # ---------------------------------------------------------------------------
    # 训练循环
    # ---------------------------------------------------------------------------
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = int(cfg["optimization"]["steps"])
    log_every = int(cfg["training"].get("log_every", 1))
    save_every = int(cfg["training"].get("save_every", 100))
    valid_every = int(cfg["training"].get("valid_every", 0))
    def _parse_deg_flag(env_key: str, cfg_key: str, fallback_key: str = "enabled") -> bool:
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no")
        if cfg_key in deg_cfg:
            return bool(deg_cfg[cfg_key])
        return bool(deg_cfg.get(fallback_key, True))

    teacher_deg = _parse_deg_flag("LTX_TEACHER_DEGRADE", "teacher_degradation")
    student_deg = _parse_deg_flag("LTX_STUDENT_DEGRADE", "student_degradation")
    deg_mode = deg_cfg.get("mode", "aigc_realbasic")
    lq_downsample_scale = tuple(deg_cfg.get("lq_downsample_scale", [2.0, 2.0]))
    sigma0 = strategy_cfg.sigma0
    velocity_loss_weight = strategy_cfg.velocity_loss_weight
    x0_loss_weight = strategy_cfg.x0_loss_weight
    fps_val = float(cfg["data"].get("fps", 24.0))

    video_decoder = None
    if valid_every > 0:
        video_decoder = VideoDecoder(cfg["model"]["model_path"], torch.bfloat16, accelerator.device)
        if accelerator.is_main_process:
            print(f"# 验证已启用: 每 {valid_every} 步, 输出 -> {output_dir / 'validation'}")

    rng = random.Random(cfg.get("seed", 42) + accelerator.process_index)
    noise_generator = torch.Generator(device=accelerator.device).manual_seed(
        cfg.get("seed", 42) + accelerator.process_index
    )

    data_iter = iter(dataloader)
    progress = tqdm(range(total_steps), disable=not accelerator.is_main_process)

    if accelerator.is_main_process:
        print(f"# 退化 (degradation): 教师={'启用' if teacher_deg else '关闭'}, 学生={'启用' if student_deg else '关闭'} (mode={deg_mode})")
        print(f"# 教师 sigma schedule: {teacher_sigma_schedule}")
        print(f"# 学生 sigma0: {sigma0}")
        print(f"# velocity_loss_weight={velocity_loss_weight}, x0_loss_weight={x0_loss_weight}")
        print(f"# 开始训练, 共 {total_steps} 步 ...")

    for global_step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(transformer):
            gt_video = batch["pixel"].to(accelerator.device, dtype=torch.bfloat16)
            captions = batch["caption"]
            batch_size = gt_video.shape[0]

            # ---- 1. 像素空间下采样 → 退化 → VAE encode → spatial upsample ----
            with torch.no_grad():
                lq_small_clean = downsample_condition_video(gt_video, lq_downsample_scale, rng)
                enc_tiling = TilingConfig.default()

                def _encode_and_upsample(lq_px: Tensor) -> Tensor:
                    lat = vae_encoder.tiled_encode(lq_px.to(torch.bfloat16), tiling_config=enc_tiling)
                    if spatial_upsampler is not None:
                        return upsample_video(latent=lat, video_encoder=vae_encoder, upsampler=spatial_upsampler)
                    ref = vae_encoder.tiled_encode(gt_video.to(torch.bfloat16), tiling_config=enc_tiling)
                    return F.interpolate(lat, size=(ref.shape[2], ref.shape[3], ref.shape[4]), mode="nearest")

                if teacher_deg == student_deg:
                    lq_small = lq_small_clean
                    if teacher_deg:
                        lq_small = build_low_quality_video(lq_small, deg_mode, rng, noise_generator)
                    lq_latent_teacher = lq_latent_student = _encode_and_upsample(lq_small)
                else:
                    lq_small_t = build_low_quality_video(lq_small_clean.clone(), deg_mode, rng, noise_generator) if teacher_deg else lq_small_clean
                    lq_small_s = build_low_quality_video(lq_small_clean.clone(), deg_mode, rng, noise_generator) if student_deg else lq_small_clean
                    lq_latent_teacher = _encode_and_upsample(lq_small_t)
                    lq_latent_student = _encode_and_upsample(lq_small_s)

                # 编码文本
                conditions = encode_prompts(captions, text_encoder, embeddings_processor)
                conditions_dict = {
                    "video_prompt_embeds": conditions["video_prompt_embeds"].to(accelerator.device),
                    "prompt_attention_mask": conditions["prompt_attention_mask"].to(accelerator.device),
                }

                # 共享噪声（教师和学生使用同一份噪声）
                noise = torch.randn(lq_latent_teacher.shape, device=accelerator.device,
                                    dtype=lq_latent_teacher.dtype, generator=noise_generator)

            # ---- 2. 教师 3 步 Euler 去噪（no_grad）→ teacher_x0 ----
            teacher_x0 = teacher_euler_denoise(
                transformer, teacher_lora, student_lora, strategy,
                lq_latent=lq_latent_teacher, noise=noise,
                sigma_schedule=teacher_sigma_schedule,
                conditions=conditions_dict, fps=fps_val,
            )

            # ---- 3. 学生单步预测（sigma=sigma0 处） ----
            # 确保学生分支激活
            student_lora.set_enabled(True)
            teacher_lora.set_enabled(False)

            num_frames = lq_latent_student.shape[2]
            height = lq_latent_student.shape[3]
            width = lq_latent_student.shape[4]
            device = lq_latent_student.device
            dtype = lq_latent_student.dtype

            # 学生输入：噪声 + 学生 lq
            x_t = sigma0 * noise + (1 - sigma0) * lq_latent_student

            # patchify → Modality → transformer forward
            video_tokens = strategy._video_patchifier.patchify(x_t)
            video_seq_len = video_tokens.shape[1]
            sigmas = torch.full((batch_size,), sigma0, device=device, dtype=dtype)
            video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)
            video_positions = strategy._get_video_positions(
                num_frames=num_frames, height=height, width=width,
                batch_size=batch_size, fps=fps_val, device=device, dtype=dtype,
            )
            video_modality = Modality(
                enabled=True, sigma=sigmas, latent=video_tokens,
                timesteps=video_timesteps, positions=video_positions,
                context=conditions_dict["video_prompt_embeds"],
                context_mask=conditions_dict["prompt_attention_mask"],
            )

            velocity_pred, _ = transformer(video=video_modality, audio=None, perturbations=None)

            # ---- 4. 计算蒸馏损失 ----
            # velocity target: v_target = noise - teacher_x0（flow matching 速度场）
            teacher_x0_tokens = strategy._video_patchifier.patchify(teacher_x0.detach())
            noise_tokens = strategy._video_patchifier.patchify(noise)
            velocity_target = noise_tokens - teacher_x0_tokens

            loss_velocity = F.mse_loss(velocity_pred, velocity_target)

            # x0 预测: student_x0 = x_t - sigma * velocity（Euler 一步到 sigma=0）
            velocity_latent = strategy._video_patchifier.unpatchify(
                velocity_pred,
                output_shape=VideoLatentShape(
                    batch=batch_size, channels=128,
                    frames=num_frames, height=height, width=width,
                ),
            )
            student_x0 = x_t - sigma0 * velocity_latent
            loss_x0 = F.mse_loss(student_x0, teacher_x0.detach())

            loss = velocity_loss_weight * loss_velocity + x0_loss_weight * loss_x0

            # ---- 4b. 像素空间 L1 + LPIPS 损失（可选） ----
            loss_pixel_l1 = torch.zeros((), device=accelerator.device)
            loss_lpips = torch.zeros((), device=accelerator.device)
            if pixel_loss_enabled and vae_decoder_nn is not None:
                num_decode = int(pixel_loss_cfg.get("num_decode_frames", 8))
                loss_pixel_l1, loss_lpips = compute_pixel_losses(
                    vae_decoder_nn, lpips_fn,
                    student_x0, teacher_x0.detach(),
                    num_frames=num_decode,
                )
                px_l1_w = float(pixel_loss_cfg.get("pixel_l1_loss_weight", 1.0))
                px_lpips_w = float(pixel_loss_cfg.get("lpips_loss_weight", 2.0))
                loss = loss + px_l1_w * loss_pixel_l1 + px_lpips_w * loss_lpips

            # ---- 5. 反向传播 + 优化器更新 ----
            accelerator.backward(loss)
            if accelerator.sync_gradients and cfg["optimization"].get("max_grad_norm", 1.0) > 0:
                accelerator.clip_grad_norm_(trainable_params, cfg["optimization"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if accelerator.is_main_process and (global_step % log_every == 0 or global_step == total_steps - 1):
            loss_val = loss.item()
            lr_val = optimizer.param_groups[0]["lr"]
            progress.set_description(
                f"loss={loss_val:.5f} v={loss_velocity.item():.5f} x0={loss_x0.item():.5f}"
            )
            if enable_swanlab:
                log_dict = {
                    "loss": loss_val, "lr": lr_val,
                    "loss_velocity": loss_velocity.item(),
                    "loss_x0": loss_x0.item(),
                }
                if pixel_loss_enabled:
                    log_dict["loss_pixel_l1"] = loss_pixel_l1.item()
                    log_dict["loss_lpips"] = loss_lpips.item()
                swanlab.log(log_dict, step=global_step)

        # ---- 保存检查点 ----
        if save_every > 0 and (global_step + 1) % save_every == 0:
            save_lora_checkpoint(
                accelerator, transformer, student_lora,
                checkpoint_dir, global_step + 1, save_metadata,
            )

        # ---- 验证（保存对比视频） ----
        if valid_every > 0 and video_decoder is not None and (global_step + 1) % valid_every == 0:
            run_validation(
                accelerator=accelerator, model=transformer, strategy=strategy,
                student_lora=student_lora, teacher_lora=teacher_lora,
                video_decoder=video_decoder,
                teacher_x0=teacher_x0, lq_latent=lq_latent_student, student_x0=student_x0,
                conditions=conditions_dict,
                output_dir=output_dir, step=global_step + 1, fps=fps_val,
            )

    # ---- 保存最终检查点 ----
    save_path = save_lora_checkpoint(
        accelerator, transformer, student_lora,
        checkpoint_dir, total_steps, save_metadata,
    )
    if accelerator.is_main_process and save_path is not None:
        print(f"# 最终检查点已保存: {save_path}")


if __name__ == "__main__":
    main()
