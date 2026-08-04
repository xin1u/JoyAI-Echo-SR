"""LTX 2.0 三步教师在线蒸馏 + DMD 分布匹配 + GAN 对抗 SR LoRA 训练脚本。

整体架构：
    ┌─────────────────────────────────────────────────────────────────┐
    │  在线蒸馏管线（继承自 train_stage2_sr_distill_v2.py）            │
    │                                                                 │
    │  GT 视频 → 下采样/退化 → VAE encode → spatial upsample → LQ latent │
    │                          ↓                                      │
    │  教师 LoRA（frozen）3步 Euler 去噪 → teacher_x0（高质量 GT）      │
    │  学生 LoRA（trainable）单步预测    → student_x0（蒸馏目标）       │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  DMD 分布匹配损失（参考 train_stage2_sr_dmd_one_step_fsdp.py）    │
    │                                                                 │
    │  student_x0 + noise → x_noisy                                   │
    │  real_score LoRA（frozen）→ pred_x0_real（真实分布得分）          │
    │  fake_score LoRA（trainable）→ pred_x0_fake（生成分布得分）       │
    │  DMD loss = ||x0_hat - (x0_hat + (real - fake)/norm)||²         │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │  GAN 对抗损失（参考 DMD2 的 cls_on_clean_image）                 │
    │                                                                 │
    │  判别器（TokenPoolingDiscriminator）：                            │
    │    real = teacher_x0 tokens → logit（应为 1）                    │
    │    fake = student_x0 tokens → logit（应为 0）                    │
    │  生成器损失：让判别器认为 student_x0 是真的                       │
    └─────────────────────────────────────────────────────────────────┘

损失函数总览：
    L_total = w_vel  * L_velocity        # 速度场 MSE（锚定教师轨迹）
            + w_x0   * L_x0              # x0 MSE（直接蒸馏）
            + w_dmd  * L_dmd             # DMD 分布匹配
            + w_reg  * L_regression      # DMD 回归正则
            + w_gan  * L_gan_generator   # GAN 生成器损失
            + w_l1   * L_pixel_l1        # 像素 L1（可选）
            + w_lpips* L_lpips           # LPIPS 感知（可选）

LoRA 分支管理（4 个 LoRA 共享 1 个 frozen base transformer）：
    - teacher:  frozen，3 步 Euler 去噪生成在线 GT
    - student:  trainable，单步预测（即 DMD 中的 generator）
    - real:     frozen，DMD real score
    - fake:     trainable，DMD fake score / critic

用法:
    python scripts/train_stage2_sr_distill_dmd_v2.py configs/ltx2_stage2_sr_distill_dmd_lora_v2.yaml

Modified for the Echo-SR release in 2026: portable configuration paths,
environment-based experiment credentials, and config-only validation.
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

# ---------------------------------------------------------------------------
# 添加项目内部包到 Python 路径
# ---------------------------------------------------------------------------
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
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn import Module
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
from ltx_sr_trainer.degradation import (
    apply_aigc_realbasicvsr_degradation,
    apply_realbasicvsr_degradation,
)
from ltx_sr_trainer.native_lora import NativeLoRAManager
from ltx_sr_trainer.training_strategies.stage2_sr_online import (
    Stage2SROnlineConfig,
    Stage2SROnlineStrategy,
)
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
    load_model as load_ltx_model,
)
from ltx_trainer.training_strategies.base_strategy import DEFAULT_FPS


# ===========================================================================
# 第一部分：配置解析与 LoRA checkpoint 工具函数
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="LTX 2.0 三步教师在线蒸馏 + DMD + GAN SR LoRA 训练"
    )
    parser.add_argument("config", type=str, help="YAML 配置文件路径")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the YAML structure without loading model weights or CUDA.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
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
        "optimization": ("learning_rate", "critic_learning_rate", "steps", "batch_size"),
        "dmd": ("generator_update_interval", "critic_update_interval", "dmd_loss_weight"),
        "gan": ("enabled", "token_dim", "hidden_dim"),
    }
    for section, keys in required.items():
        values = cfg.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Missing or invalid config section: {section}")
        missing = [key for key in keys if key not in values or values[key] in (None, "", [])]
        if missing:
            raise ValueError(f"Missing required keys in {section}: {', '.join(missing)}")


def _normalize_path_string(value: str | Path | None) -> str | Path | None:
    """去除路径字符串两端的引号和空格。"""
    if value is None or not isinstance(value, str):
        return value
    return value.strip().strip("\"'")


def normalize_config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """清理配置中所有路径字段的引号和空格。"""
    cfg = dict(cfg)
    for section, keys in {
        "model": ("model_path", "text_encoder_path", "init_lora_path", "spatial_upsampler_path"),
        "training": ("output_dir",),
        "dmd": ("real_lora_path", "fake_lora_init_path"),
    }.items():
        if section in cfg:
            cfg[section] = dict(cfg[section])
            for key in keys:
                if key in cfg[section]:
                    cfg[section][key] = _normalize_path_string(cfg[section].get(key))
    return cfg


def inspect_lora_checkpoint(path: str | Path) -> dict[str, Any]:
    """读取 LoRA checkpoint，推断 rank / alpha / target_modules。"""
    path = Path(_normalize_path_string(path)).expanduser().resolve()
    rank: int | None = None
    alpha: int | None = None
    module_ranks: dict[str, int] = {}
    unique_ranks: set[int] = set()
    alpha_from_metadata = False
    pattern = re.compile(r"^diffusion_model\.(.+)\.lora_[AB]\.weight$")
    target_set: set[str] = set()

    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        if "lora_alpha" in metadata:
            try:
                alpha = int(metadata["lora_alpha"])
                alpha_from_metadata = True
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
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "target_module_count": len(target_modules),
        "module_ranks": module_ranks,
        "module_alphas": {m: alpha for m in target_modules},
        "unique_ranks": sorted(unique_ranks),
        "path": str(path),
        "alpha_from_metadata": alpha_from_metadata,
    }


def maybe_sync_lora_hparams_from_init(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """若配置了 init_lora_path，从中读取并覆盖 LoRA 超参。"""
    init_lora_path = cfg.get("model", {}).get("init_lora_path")
    if not init_lora_path:
        return cfg, None
    inferred = inspect_lora_checkpoint(init_lora_path)
    if not inferred.get("alpha_from_metadata") and cfg.get("lora", {}).get("alpha") is not None:
        inferred["alpha"] = int(cfg["lora"]["alpha"])
        inferred["module_alphas"] = {m: inferred["alpha"] for m in inferred["target_modules"]}
    cfg = dict(cfg)
    cfg["lora"] = dict(cfg["lora"])
    cfg["lora"]["rank"] = inferred["rank"]
    cfg["lora"]["alpha"] = inferred["alpha"]
    cfg["lora"]["target_modules"] = inferred["target_modules"]
    return cfg, inferred


def build_lora_metadata(cfg: dict[str, Any], inferred: dict[str, Any] | None = None) -> dict[str, str]:
    """构建保存 LoRA checkpoint 时附带的 metadata 信息。"""
    metadata = {
        "stage2_sr_distill_dmd": "true",
        "model_version": "ltx-2.0-19b",
        "stage2_noise_sigma": str(cfg["training"].get("sigma0", 0.909375)),
        "dmd_sigma_min": str(cfg.get("dmd", {}).get("sigma_min", 0.02)),
        "dmd_sigma_max": str(cfg.get("dmd", {}).get("sigma_max", 0.98)),
        "lora_rank": str(cfg["lora"].get("rank", 32)),
        "lora_alpha": str(cfg["lora"].get("alpha", 32)),
        "lora_target_modules": json.dumps(cfg["lora"]["target_modules"]),
    }
    if inferred is not None:
        metadata["init_lora_path"] = inferred["path"]
        metadata["lora_spec_source"] = "init_lora_checkpoint"
    return metadata


# ===========================================================================
# 第二部分：文本编码
# ===========================================================================


def encode_prompts(
    prompts: list[str],
    text_encoder: Any,
    embeddings_processor: Any,
) -> dict[str, Tensor | None]:
    """对文本 prompt 进行编码，提取 video embedding（无音频分支）。"""
    video_embeddings, attention_masks = [], []
    with torch.no_grad():
        for prompt in prompts:
            hidden_states, attention_mask = text_encoder.encode(prompt)
            out = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
            video_embeddings.append(out.video_encoding)
            attention_masks.append(out.attention_mask)
    return {
        "video_prompt_embeds": torch.cat(video_embeddings, dim=0),
        "prompt_attention_mask": torch.cat(attention_masks, dim=0),
    }


# ===========================================================================
# 第三部分：LoRA 分支管理（4 分支共享 base transformer）
# ===========================================================================


def get_unwrapped_model(model: Any) -> Any:
    """获取 accelerate/DDP 包装下的原始模型。"""
    return model.module if hasattr(model, "module") else model


def get_lora_manager(model: Any, name: str) -> NativeLoRAManager:
    """根据名称获取 LoRA manager。"""
    unwrapped = get_unwrapped_model(model)
    return getattr(unwrapped, name)


def set_active_branch(model: Any, active: str) -> None:
    """切换 4 分支 LoRA 中的活跃分支。

    4 个分支功能：
      - teacher:  在线 3 步 Euler 去噪（frozen，推理生成 GT）
      - student:  单步蒸馏预测（trainable，即 DMD 中的 generator）
      - real:     DMD real score（frozen，提供真实分布梯度信号）
      - fake:     DMD fake score / critic（trainable，拟合生成分布）

    同一时刻只能激活一个分支，通过 hook enable/disable 切换。
    """
    branch_names = ("teacher", "student", "real", "fake")
    manager_attr_names = {
        "teacher": "_lora_teacher",
        "student": "_lora_student",
        "real": "_lora_real",
        "fake": "_lora_fake",
    }
    if active not in branch_names:
        raise ValueError(f"未知 LoRA 分支 {active!r}; 期望: {branch_names}")
    for name in branch_names:
        mgr = get_lora_manager(model, manager_attr_names[name])
        mgr.set_enabled(name == active)


# ===========================================================================
# 第四部分：退化 + VAE 编码
# ===========================================================================


def build_low_quality_video(
    video: Tensor,
    mode: str,
    rng: random.Random,
    generator: torch.Generator | None = None,
) -> Tensor:
    """对视频施加退化处理，生成低质量输入。"""
    if generator is None:
        generator = torch.Generator(device=video.device).manual_seed(rng.randint(0, 2**31 - 1))
    if mode == "aigc_realbasic":
        return apply_aigc_realbasicvsr_degradation(video, rng, generator)
    if mode in ("realbasicvsr", "realesrgan"):
        return apply_realbasicvsr_degradation(video, rng, generator)
    raise ValueError(f"未知退化模式: {mode}")


def downsample_condition_video(
    video: Tensor,
    scale_range: tuple[float, float],
    rng: random.Random,
) -> Tensor:
    """像素空间下采样。"""
    factor = rng.uniform(*scale_range)
    scale = 1.0 / max(factor, 1e-8)
    if scale >= 0.999:
        return video
    b, c, t, h, w = video.shape
    target_h = max(1, int(round(h * scale)))
    target_w = max(1, int(round(w * scale)))
    if target_h == h and target_w == w:
        return video
    video_2d = rearrange(video, "b c t h w -> (b t) c h w").contiguous().float()
    resized = F.interpolate(video_2d, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return rearrange(resized.to(dtype=video.dtype), "(b t) c h w -> b c t h w", b=b, t=t)


# ===========================================================================
# 第五部分：教师 3 步 Euler 去噪
# ===========================================================================


@torch.no_grad()
def teacher_euler_denoise(
    transformer: Any,
    model: Any,
    strategy: Stage2SROnlineStrategy,
    *,
    lq_latent: Tensor,
    noise: Tensor,
    sigma_schedule: list[float],
    conditions: dict[str, Tensor | None],
    fps: float,
) -> Tensor:
    """教师 3 步 Euler 去噪：从 lq_latent + noise 出发，逐步去噪到 sigma=0。

    数学公式：
        x_t = sigma_0 * noise + (1 - sigma_0) * lq_latent
        x_{t-1} = x_t + (sigma_{next} - sigma_{cur}) * velocity
    """
    set_active_branch(model, "teacher")

    batch_size = lq_latent.shape[0]
    num_frames = lq_latent.shape[2]
    height = lq_latent.shape[3]
    width = lq_latent.shape[4]
    device = lq_latent.device
    dtype = lq_latent.dtype

    video_prompt_embeds = conditions["video_prompt_embeds"]
    prompt_attention_mask = conditions["prompt_attention_mask"]

    x_t = sigma_schedule[0] * noise + (1 - sigma_schedule[0]) * lq_latent

    for i in range(len(sigma_schedule) - 1):
        sigma_cur = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]

        video_tokens = strategy._video_patchifier.patchify(x_t)
        video_seq_len = video_tokens.shape[1]
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

        velocity, _ = transformer(video=video_modality, audio=None, perturbations=None)
        velocity_latent = strategy._video_patchifier.unpatchify(
            velocity,
            output_shape=VideoLatentShape(
                batch=batch_size, channels=128,
                frames=num_frames, height=height, width=width,
            ),
        )
        x_t = x_t + (sigma_next - sigma_cur) * velocity_latent

    return x_t


# ===========================================================================
# 第六部分：前向推理辅助函数（用于 DMD 各分支）
# ===========================================================================


def make_video_modality(
    *,
    strategy: Stage2SROnlineStrategy,
    video_tokens: Tensor,
    sigma: Tensor,
    conditions: dict[str, Tensor],
    num_frames: int,
    height: int,
    width: int,
    fps: float,
) -> Modality:
    """构建 transformer 前向所需的 Modality 对象。"""
    batch_size, seq_len = video_tokens.shape[:2]
    device = video_tokens.device
    dtype = video_tokens.dtype
    sigma = sigma.to(device=device, dtype=dtype).view(-1)
    timesteps = sigma.view(-1, 1).expand(-1, seq_len)
    positions = strategy._get_video_positions(
        num_frames=num_frames, height=height, width=width,
        batch_size=batch_size, fps=fps, device=device, dtype=dtype,
    )
    return Modality(
        enabled=True, sigma=sigma, latent=video_tokens,
        timesteps=timesteps, positions=positions,
        context=conditions["video_prompt_embeds"],
        context_mask=conditions["prompt_attention_mask"],
    )


def forward_video_velocity(
    model: Any,
    *,
    branch: str,
    strategy: Stage2SROnlineStrategy,
    video_tokens: Tensor,
    sigma: Tensor,
    conditions: dict[str, Tensor],
    shape_meta: dict[str, int | float],
) -> Tensor:
    """切换到指定 LoRA 分支并执行前向推理，返回 velocity 预测。"""
    set_active_branch(model, branch)
    modality = make_video_modality(
        strategy=strategy, video_tokens=video_tokens, sigma=sigma,
        conditions=conditions,
        num_frames=int(shape_meta["num_frames"]),
        height=int(shape_meta["height"]),
        width=int(shape_meta["width"]),
        fps=float(shape_meta["fps"]),
    )
    video_pred, _ = model(video=modality, audio=None, perturbations=None)
    return video_pred


# ===========================================================================
# 第七部分：Flow Matching 工具函数
# ===========================================================================


def velocity_to_x0(x_t: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """从 velocity 预测反推 x0: x0 = x_t - sigma * velocity。"""
    while sigma.ndim < x_t.ndim:
        sigma = sigma.unsqueeze(-1)
    return x_t - sigma.to(dtype=x_t.dtype) * velocity


def flow_match_noise(x0: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Flow matching 加噪: x_t = (1 - sigma) * x0 + sigma * noise。"""
    while sigma.ndim < x0.ndim:
        sigma = sigma.unsqueeze(-1)
    sigma = sigma.to(dtype=x0.dtype)
    return (1 - sigma) * x0 + sigma * noise


def sample_sigmas(batch_size: int, *, device: torch.device, dtype: torch.dtype, cfg: dict[str, Any]) -> Tensor:
    """采样 DMD 重加噪 sigma 值。

    支持多种采样策略：
    - uniform: 均匀采样
    - logit_normal: logit 正态分布
    - beta_low: Beta 分布（偏向低 sigma）
    - teacher_sigmas: 从教师 sigma schedule 中采样
    """
    dmd_cfg = cfg.get("dmd", {})
    sigma_min = float(dmd_cfg.get("sigma_min", 0.02))
    sigma_max = float(dmd_cfg.get("sigma_max", 0.98))
    mode = dmd_cfg.get("sigma_sampling", "uniform")

    if mode == "uniform":
        u = torch.rand(batch_size, device=device, dtype=dtype)
        return sigma_min + (sigma_max - sigma_min) * u
    if mode == "logit_normal":
        mean = float(dmd_cfg.get("logit_mean", 0.0))
        std = float(dmd_cfg.get("logit_std", 1.0))
        u = torch.randn(batch_size, device=device, dtype=dtype) * std + mean
        s = torch.sigmoid(u)
        return sigma_min + (sigma_max - sigma_min) * s
    if mode == "beta_low":
        alpha_param = float(dmd_cfg.get("beta_alpha", 0.5))
        beta_param = float(dmd_cfg.get("beta_beta", 1.0))
        dist = torch.distributions.Beta(alpha_param, beta_param)
        u = dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return sigma_min + (sigma_max - sigma_min) * u
    if mode == "teacher_sigmas":
        default_sigmas = [0.909375, 0.725, 0.421875]
        sigma_list = dmd_cfg.get("teacher_sigma_values", default_sigmas)
        sigma_pool = torch.tensor(sigma_list, device=device, dtype=dtype)
        indices = torch.randint(0, len(sigma_pool), (batch_size,), device=device)
        return sigma_pool[indices]
    raise ValueError(f"不支持的 dmd.sigma_sampling: {mode}")


# ===========================================================================
# 第八部分：DMD 损失计算
# ===========================================================================


def compute_dmd_loss(
    model: Any,
    *,
    strategy: Stage2SROnlineStrategy,
    student_x0_tokens: Tensor,
    conditions: dict[str, Tensor],
    shape_meta: dict[str, int | float],
    cfg: dict[str, Any],
    normalizer_ema: dict[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """计算 DMD 分布匹配损失。

    核心步骤：
    1. 对 student_x0 重加噪 → x_noisy
    2. real score（frozen）预测 x0_real
    3. fake score（trainable）预测 x0_fake
    4. DMD 梯度 = (x0_real - x0_fake) / normalizer
    5. 损失 = ||x0_hat - (x0_hat + DMD_grad)||²

    梯度只流回 student（generator），real/fake 分支在 no_grad 下运行。
    """
    dmd_cfg = cfg.get("dmd", {})

    # 采样 sigma 并对 student 预测结果重加噪
    sigmas = sample_sigmas(
        student_x0_tokens.shape[0],
        device=student_x0_tokens.device,
        dtype=student_x0_tokens.dtype,
        cfg=cfg,
    )
    noise = torch.randn_like(student_x0_tokens)
    x_noisy = flow_match_noise(student_x0_tokens, noise, sigmas)

    # Real score 和 Fake score 前向（no_grad，不计算梯度）
    with torch.no_grad():
        v_real = forward_video_velocity(
            model, branch="real", strategy=strategy,
            video_tokens=x_noisy, sigma=sigmas,
            conditions=conditions, shape_meta=shape_meta,
        )
        pred_x0_real = velocity_to_x0(x_noisy, v_real, sigmas).float()

        v_fake = forward_video_velocity(
            model, branch="fake", strategy=strategy,
            video_tokens=x_noisy, sigma=sigmas,
            conditions=conditions, shape_meta=shape_meta,
        )
        pred_x0_fake = velocity_to_x0(x_noisy, v_fake, sigmas).float()

    # 恢复 student 分支（确保后续梯度检查点重计算时分支正确）
    set_active_branch(model, "student")

    # 计算 DMD 梯度更新方向 + 归一化
    reduce_dims = tuple(range(1, student_x0_tokens.ndim))
    eps = float(dmd_cfg.get("normalizer_eps", 1.0e-6))
    per_sample_norm = (pred_x0_real - student_x0_tokens.detach().float()).abs().mean(
        dim=reduce_dims, keepdim=True
    )

    # 可选 EMA 归一化器（稳定不同 step 间的梯度幅度）
    ema_decay = float(dmd_cfg.get("normalizer_ema_decay", 0.0))
    if ema_decay > 0 and normalizer_ema is not None:
        batch_mean = per_sample_norm.mean().detach()
        if not normalizer_ema["initialized"]:
            normalizer_ema["value"] = batch_mean
            normalizer_ema["initialized"] = True
        else:
            normalizer_ema["value"] = ema_decay * normalizer_ema["value"] + (1 - ema_decay) * batch_mean
        ema_val = normalizer_ema["value"]
        normalizer = per_sample_norm.clamp(min=0.5 * ema_val, max=2.0 * ema_val)
    else:
        normalizer = per_sample_norm

    normalizer = normalizer.clamp_min(eps)
    dm_update = (pred_x0_real - pred_x0_fake) / normalizer
    dm_update = torch.nan_to_num(dm_update)

    # DMD 损失：将 DMD 梯度方向作为伪目标
    loss_dmd = 0.5 * F.mse_loss(
        student_x0_tokens.float(),
        (student_x0_tokens.float() + dm_update).detach(),
        reduction="mean",
    )

    logs = {
        "loss_dmd_dm": loss_dmd.detach(),
        "dmd_sigma_mean": sigmas.float().mean().detach(),
        "dmd_normalizer": normalizer.float().mean().detach(),
        "dmd_update_abs": dm_update.float().abs().mean().detach(),
    }
    return loss_dmd, logs


def compute_fake_score_critic_loss(
    model: Any,
    *,
    strategy: Stage2SROnlineStrategy,
    student_x0_tokens: Tensor,
    conditions: dict[str, Tensor],
    shape_meta: dict[str, int | float],
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """计算 fake score critic 的 flow matching 损失。

    Critic 学习拟合 student 生成分布上的 velocity 场：
        target_velocity = noise - x0_hat
        loss = ||v_fake(x_noisy, sigma) - target_velocity||²
    """
    sigmas = sample_sigmas(
        student_x0_tokens.shape[0],
        device=student_x0_tokens.device,
        dtype=student_x0_tokens.dtype,
        cfg=cfg,
    )
    noise = torch.randn_like(student_x0_tokens)
    x_noisy = flow_match_noise(student_x0_tokens, noise, sigmas)

    v_fake = forward_video_velocity(
        model, branch="fake", strategy=strategy,
        video_tokens=x_noisy, sigma=sigmas,
        conditions=conditions, shape_meta=shape_meta,
    )
    # 保持 fake 分支激活（供梯度检查点重计算）
    set_active_branch(model, "fake")

    target_velocity = noise - student_x0_tokens
    loss = F.mse_loss(v_fake.float(), target_velocity.float().detach(), reduction="mean")

    return loss, {
        "loss_critic": loss.detach(),
        "critic_sigma_mean": sigmas.float().mean().detach(),
    }


# ===========================================================================
# 第九部分：GAN 判别器 + 损失
# ===========================================================================


class TokenPoolingDiscriminator(Module):
    """轻量级 token-pooling 判别器（参考 DMD2 cls_on_clean_image）。

    架构：对 patchified latent tokens 做 mean-pool → MLP → scalar logit。
    输入：[B, seq_len, token_dim] 的 token 序列
    输出：[B] 的 realism logit（值越大表示越像真实样本）
    """

    def __init__(self, token_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(token_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        pooled = tokens.float().mean(dim=1)
        return self.net(pooled).squeeze(-1)


def compute_gan_discriminator_loss(
    discriminator: TokenPoolingDiscriminator,
    real_tokens: Tensor,
    fake_tokens: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """判别器 non-saturating GAN 损失。

    real_tokens: 教师 x0（真实样本），fake_tokens: 学生 x0（生成样本）。
    两者均 detach，判别器梯度不传回生成器。
    """
    pred_real = discriminator(real_tokens.detach())
    pred_fake = discriminator(fake_tokens.detach())
    loss = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
    return loss, {
        "gan_d_loss": loss.detach(),
        "gan_pred_real_mean": torch.sigmoid(pred_real).mean().detach(),
        "gan_pred_fake_mean": torch.sigmoid(pred_fake).mean().detach(),
    }


def compute_gan_generator_loss(
    discriminator: TokenPoolingDiscriminator,
    fake_tokens: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """生成器 non-saturating GAN 损失。

    fake_tokens 保留梯度，使生成器学习欺骗判别器。
    """
    pred_fake = discriminator(fake_tokens)
    loss = F.softplus(-pred_fake).mean()
    return loss, {"gan_g_loss": loss.detach()}


# ===========================================================================
# 第十部分：像素空间辅助损失
# ===========================================================================


def compute_pixel_losses(
    vae_decoder: Module,
    lpips_fn: Module,
    pred_x0_latent: Tensor,
    gt_latent: Tensor,
    num_frames: int = 8,
) -> tuple[Tensor, Tensor]:
    """随机抽取帧 → VAE decode → 像素空间 L1 + LPIPS 损失。"""
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


# ===========================================================================
# 第十一部分：LoRA 构建 + checkpoint 保存
# ===========================================================================


def attach_all_loras(
    transformer: Any,
    cfg: dict[str, Any],
    inferred_init_lora: dict[str, Any] | None,
    is_fsdp: bool,
) -> tuple[NativeLoRAManager, NativeLoRAManager, NativeLoRAManager, NativeLoRAManager]:
    """创建 4 个 LoRA 分支并挂载到 transformer 上。

    返回 (teacher_lora, student_lora, real_lora, fake_lora)。
    """
    module_ranks = inferred_init_lora["module_ranks"] if inferred_init_lora else None
    module_alphas = inferred_init_lora["module_alphas"] if inferred_init_lora else None

    param_dtype_str = str(cfg["lora"].get("param_dtype", "bf16")).lower()
    param_dtype = {
        "fp32": torch.float32, "float32": torch.float32,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
    }.get(param_dtype_str)
    if param_dtype is None:
        raise ValueError(f"不支持的 lora.param_dtype={param_dtype_str!r}")

    common_kwargs = dict(
        target_modules=cfg["lora"]["target_modules"],
        rank=cfg["lora"].get("rank", 32),
        alpha=cfg["lora"].get("alpha", 32),
        module_ranks=module_ranks,
        module_alphas=module_alphas,
        dropout=cfg["lora"].get("dropout", 0.0),
        param_dtype=param_dtype,
        attach_to_targets=is_fsdp,
    )

    # ---- 创建 4 个 LoRA 分支 ----
    teacher_lora = NativeLoRAManager(transformer, namespace="teacher", **common_kwargs)
    student_lora = NativeLoRAManager(transformer, namespace="student", **common_kwargs)
    real_lora = NativeLoRAManager(transformer, namespace="real", **common_kwargs)
    fake_lora = NativeLoRAManager(transformer, namespace="fake", **common_kwargs)

    # 注册为 transformer 的子模块（让 accelerate/FSDP 能感知参数）
    transformer._lora_teacher = teacher_lora
    transformer._lora_student = student_lora
    transformer._lora_real = real_lora
    transformer._lora_fake = fake_lora
    # 兼容 save 工具
    transformer._native_lora_manager = student_lora

    # ---- 加载初始权重 ----
    init_lora_path = cfg["model"].get("init_lora_path")
    dmd_cfg = cfg.get("dmd", {})

    # teacher 和 student 均从 init_lora_path 初始化
    if init_lora_path:
        teacher_lora.load_lora_weights(init_lora_path)
        student_lora.load_lora_weights(init_lora_path)

    # real score: 优先用 dmd.real_lora_path，否则用 init_lora_path
    real_path = dmd_cfg.get("real_lora_path") or init_lora_path
    if real_path:
        real_lora.load_lora_weights(real_path)

    # fake score: 优先用 dmd.fake_lora_init_path，否则用 init_lora_path
    fake_path = dmd_cfg.get("fake_lora_init_path") or init_lora_path
    if fake_path:
        fake_lora.load_lora_weights(fake_path)

    # ---- 设置可训练性 ----
    teacher_lora.requires_grad_(False)  # 教师: frozen
    real_lora.requires_grad_(False)     # real score: frozen
    student_lora.requires_grad_(True)   # 学生/generator: trainable
    fake_lora.requires_grad_(True)      # fake score/critic: trainable

    # 冻结学生 LoRA 中的 audio 相关参数（LTX 2.0 纯视频，不训练 audio 分支）
    for name, param in student_lora.named_lora_parameters():
        if "audio" in name:
            param.requires_grad_(False)
    for name, param in fake_lora.named_lora_parameters():
        if "audio" in name:
            param.requires_grad_(False)

    # 默认激活 student 分支
    teacher_lora.set_enabled(False)
    student_lora.set_enabled(True)
    real_lora.set_enabled(False)
    fake_lora.set_enabled(False)

    return teacher_lora, student_lora, real_lora, fake_lora


def save_student_lora_checkpoint(
    accelerator: Accelerator,
    model: Any,
    output_dir: Path,
    step: int,
    metadata: dict[str, str],
) -> Path | None:
    """保存学生 LoRA 权重（即推理使用的最终产物）。"""
    accelerator.wait_for_everyone()
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    fsdp_state_dict = accelerator.get_state_dict(model) if is_fsdp else None
    if not accelerator.is_main_process:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
    manager = getattr(unwrapped, "_lora_student")
    lora_state = {
        k: v.to(torch.bfloat16) if isinstance(v, Tensor) else v
        for k, v in manager.export_state_dict(model_state_dict=fsdp_state_dict).items()
    }
    save_path = output_dir / f"lora_weights_step_{step:05d}.safetensors"
    save_file(lora_state, save_path, metadata=metadata)
    return save_path


def save_fake_lora_checkpoint(
    accelerator: Accelerator,
    model: Any,
    output_dir: Path,
    step: int,
    metadata: dict[str, str],
) -> Path | None:
    """保存 fake score LoRA 权重（调试/恢复用）。"""
    accelerator.wait_for_everyone()
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    fsdp_state_dict = accelerator.get_state_dict(model) if is_fsdp else None
    if not accelerator.is_main_process:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
    manager = getattr(unwrapped, "_lora_fake")
    lora_state = {
        k: v.to(torch.bfloat16) if isinstance(v, Tensor) else v
        for k, v in manager.export_state_dict(model_state_dict=fsdp_state_dict).items()
    }
    save_path = output_dir / f"fake_score_lora_step_{step:05d}.safetensors"
    save_file(lora_state, save_path, metadata={**metadata, "fake_score_lora": "true"})
    return save_path


# ===========================================================================
# 第十二部分：日志与指标
# ===========================================================================


def append_metrics_log(output_dir: Path, step: int, metrics: dict[str, float]) -> None:
    """将训练指标追加写入 JSONL 和 CSV 文件。"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    record = {"step": int(step), **{k: float(v) for k, v in sorted(metrics.items())}}
    jsonl_path = log_dir / "train_metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = log_dir / "train_metrics.csv"
    if not csv_path.exists():
        keys = list(record.keys())
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
    else:
        with csv_path.open("r", encoding="utf-8") as f:
            keys = f.readline().strip().split(",")
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(",".join(str(record.get(k, "")) for k in keys) + "\n")


# ===========================================================================
# 第十三部分：验证
# ===========================================================================


def _materialize_video_iterator(video_iter: Any) -> Tensor:
    """将 VideoDecoder 的 chunk 迭代器拼接为完整 tensor。"""
    chunks = list(video_iter)
    if not chunks:
        raise RuntimeError("Video decoder returned no chunks")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def run_validation(
    *,
    accelerator: Accelerator,
    model: Any,
    strategy: Stage2SROnlineStrategy,
    video_decoder: VideoDecoder,
    teacher_x0: Tensor,
    lq_latent: Tensor,
    student_x0: Tensor,
    output_dir: Path,
    step: int,
    fps: float,
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
    encode_video(video=sr_video, fps=fps_int, audio=None,
                 output_path=str(val_dir / "student_sr.mp4"), video_chunks_number=1)
    encode_video(video=teacher_video, fps=fps_int, audio=None,
                 output_path=str(val_dir / "teacher_gt.mp4"), video_chunks_number=1)
    encode_video(video=lq_video, fps=fps_int, audio=None,
                 output_path=str(val_dir / "lq.mp4"), video_chunks_number=1)

    min_f = min(sr_video.shape[0], teacher_video.shape[0], lq_video.shape[0])
    comparison = torch.cat([lq_video[:min_f], sr_video[:min_f], teacher_video[:min_f]], dim=2)
    encode_video(video=comparison, fps=fps_int, audio=None,
                 output_path=str(val_dir / "compare_lq_vs_student_vs_teacher.mp4"),
                 video_chunks_number=1)


# ===========================================================================
# 第十四部分：主训练函数
# ===========================================================================


def main() -> None:
    args = parse_args()
    cfg = normalize_config_paths(load_config(args.config))
    validate_config_structure(cfg)
    if args.validate_config:
        print(f"Config OK: {Path(args.config).resolve()}")
        return
    cfg, inferred_init_lora = maybe_sync_lora_hparams_from_init(cfg)

    # ==================== 初始化 Accelerator ====================
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

    # ==================== SwanLab 日志 ====================
    swanlab_cfg = cfg.get("swanlab", {})
    enable_swanlab = bool(swanlab_cfg.get("enable_swanlab", False)) and swanlab is not None
    if enable_swanlab and accelerator.is_main_process:
        swanlab_api_key = os.environ.get("SWANLAB_API_KEY")
        if swanlab_api_key:
            swanlab.login(api_key=str(swanlab_api_key))
        exp_name = str(swanlab_cfg.get("swanlab_experiment_name", "")) or Path(cfg["training"]["output_dir"]).name
        swanlab.init(
            project=str(swanlab_cfg.get("swanlab_project", "LTX-2.0-stage2-sr-distill-dmd")),
            experiment_name=exp_name, config=cfg,
        )

    # ==================== 训练策略 ====================
    deg_cfg = cfg.get("degradation", {})
    strategy_cfg = Stage2SROnlineConfig(
        sigma0=cfg["training"].get("sigma0", 0.909375),
        velocity_loss_weight=cfg["training"].get("velocity_loss_weight", 1.0),
        x0_loss_weight=cfg["training"].get("x0_loss_weight", 0.25),
        condition_noise_min=0.0, condition_noise_max=0.0,
        condition_drop_prob_min=0.0, condition_drop_prob_max=0.0,
    )
    strategy = Stage2SROnlineStrategy(strategy_cfg)
    save_metadata = build_lora_metadata(cfg, inferred_init_lora)

    # ==================== 教师 sigma schedule ====================
    teacher_cfg = cfg.get("teacher", {})
    teacher_sigma_schedule = teacher_cfg.get("sigma_schedule", [0.909375, 0.725, 0.421875, 0.0])

    # ==================== 数据集 ====================
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

    # ==================== 加载 Transformer + 4 路 LoRA ====================
    components = load_ltx_model(
        checkpoint_path=cfg["model"]["model_path"],
        device="cpu", dtype=torch.bfloat16,
        with_video_vae_encoder=False, with_video_vae_decoder=False,
        with_audio_vae_decoder=False, with_vocoder=False, with_text_encoder=False,
    )
    transformer = components.transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)

    teacher_lora, student_lora, real_lora, fake_lora = attach_all_loras(
        transformer, cfg, inferred_init_lora, is_fsdp
    )

    transformer.set_gradient_checkpointing(cfg["optimization"].get("enable_gradient_checkpointing", True))
    transformer.train()

    # ==================== 优化器 ====================
    # 学生/生成器优化器
    student_params = [p for p in student_lora.parameters() if p.requires_grad]
    student_optimizer = AdamW(
        student_params,
        lr=cfg["optimization"].get("learning_rate", 1e-5),
        betas=tuple(cfg["optimization"].get("adam_betas", [0.9, 0.95])),
        weight_decay=cfg["optimization"].get("weight_decay", 0.0),
        eps=cfg["optimization"].get("adam_eps", 1e-8),
        foreach=cfg["optimization"].get("adam_foreach", False),
    )

    # Fake score / critic 优化器
    dmd_cfg = cfg.get("dmd", {})
    critic_params = [p for p in fake_lora.parameters() if p.requires_grad]
    critic_optimizer = AdamW(
        critic_params,
        lr=cfg["optimization"].get("critic_learning_rate", dmd_cfg.get("critic_learning_rate", 5e-6)),
        betas=tuple(cfg["optimization"].get("critic_adam_betas", cfg["optimization"].get("adam_betas", [0.9, 0.95]))),
        weight_decay=cfg["optimization"].get("weight_decay", 0.0),
        eps=cfg["optimization"].get("adam_eps", 1e-8),
        foreach=cfg["optimization"].get("adam_foreach", False),
    )

    # GAN 判别器（可选）
    gan_cfg = cfg.get("gan", {})
    gan_enabled = bool(gan_cfg.get("enabled", False))
    discriminator = None
    discriminator_optimizer = None
    if gan_enabled:
        token_dim = int(gan_cfg.get("token_dim", 128))
        hidden_dim = int(gan_cfg.get("hidden_dim", 512))
        discriminator = TokenPoolingDiscriminator(token_dim=token_dim, hidden_dim=hidden_dim)
        discriminator.train()
        discriminator_optimizer = AdamW(
            discriminator.parameters(),
            lr=float(gan_cfg.get("discriminator_lr", 5e-5)),
            betas=tuple(cfg["optimization"].get("adam_betas", [0.9, 0.95])),
            weight_decay=cfg["optimization"].get("weight_decay", 0.0),
            eps=cfg["optimization"].get("adam_eps", 1e-8),
        )

    # ==================== 像素损失模块（可选） ====================
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

    # ==================== Accelerate prepare ====================
    prepare_args = [transformer, student_optimizer, critic_optimizer, dataloader]
    if discriminator is not None:
        prepare_args.extend([discriminator, discriminator_optimizer])
    prepared = accelerator.prepare(*prepare_args)
    transformer, student_optimizer, critic_optimizer, dataloader = prepared[:4]
    if discriminator is not None:
        discriminator, discriminator_optimizer = prepared[4], prepared[5]

    # ==================== VAE encoder + spatial upsampler ====================
    vae_encoder = load_video_vae_encoder(
        checkpoint_path=cfg["model"]["model_path"],
        device=accelerator.device, dtype=torch.bfloat16,
    )
    vae_encoder.requires_grad_(False)
    vae_encoder.eval()

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

    # ==================== 文本编码器 ====================
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

    # ==================== 打印训练信息 ====================
    if accelerator.is_main_process:
        student_count = sum(p.numel() for p in student_params)
        critic_count = sum(p.numel() for p in critic_params)
        teacher_count = sum(p.numel() for p in teacher_lora.parameters())
        real_count = sum(p.numel() for p in real_lora.parameters())
        print("=" * 60)
        print("  LTX 2.0 在线蒸馏 + DMD + GAN SR LoRA 训练")
        print("=" * 60)
        print(f"  分布式类型: {accelerator.distributed_type}")
        print(f"  LoRA 参数量:")
        print(f"    student (trainable): {student_count:,}")
        print(f"    fake_score (trainable): {critic_count:,}")
        print(f"    teacher (frozen): {teacher_count:,}")
        print(f"    real_score (frozen): {real_count:,}")
        print(f"  LoRA rank={cfg['lora']['rank']}, alpha={cfg['lora']['alpha']}")
        print(f"  DMD sigma range: [{dmd_cfg.get('sigma_min', 0.02)}, {dmd_cfg.get('sigma_max', 0.98)}]")
        print(f"  DMD sigma_sampling: {dmd_cfg.get('sigma_sampling', 'uniform')}")
        if gan_enabled:
            d_params = sum(p.numel() for p in discriminator.parameters())
            print(f"  GAN 判别器: params={d_params:,}, "
                  f"gen_weight={gan_cfg.get('generator_loss_weight', 0.1)}, "
                  f"disc_weight={gan_cfg.get('discriminator_loss_weight', 1.0)}")
        else:
            print("  GAN 判别器: 未启用")
        if pixel_loss_enabled:
            print(f"  像素损失: L1 weight={pixel_loss_cfg.get('pixel_l1_loss_weight', 1.0)}, "
                  f"LPIPS weight={pixel_loss_cfg.get('lpips_loss_weight', 2.0)}")
        print("=" * 60)

    # ==================== 训练超参 ====================
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = int(cfg["optimization"]["steps"])
    log_every = int(cfg["training"].get("log_every", 1))
    save_every = int(cfg["training"].get("save_every", 100))
    valid_every = int(cfg["training"].get("valid_every", 0))
    max_grad_norm = float(cfg["optimization"].get("max_grad_norm", 1.0))

    # DMD 训练节奏
    gen_interval = int(dmd_cfg.get("generator_update_interval", 5))
    critic_interval = int(dmd_cfg.get("critic_update_interval", 1))
    save_fake = bool(dmd_cfg.get("save_fake_score_lora", False))

    # 损失权重
    sigma0 = strategy_cfg.sigma0
    velocity_loss_weight = float(cfg["training"].get("velocity_loss_weight", 1.0))
    x0_loss_weight = float(cfg["training"].get("x0_loss_weight", 0.25))
    dmd_loss_weight = float(dmd_cfg.get("dmd_loss_weight", 1.0))
    regression_loss_weight = float(dmd_cfg.get("regression_loss_weight", 0.0))

    # 退化设置
    def _parse_deg_flag(env_key: str, cfg_key: str) -> bool:
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no")
        if cfg_key in deg_cfg:
            return bool(deg_cfg[cfg_key])
        return bool(deg_cfg.get("enabled", True))

    teacher_deg = _parse_deg_flag("LTX_TEACHER_DEGRADE", "teacher_degradation")
    student_deg = _parse_deg_flag("LTX_STUDENT_DEGRADE", "student_degradation")
    deg_mode = deg_cfg.get("mode", "aigc_realbasic")
    lq_downsample_scale = tuple(deg_cfg.get("lq_downsample_scale", [2.0, 2.0]))
    fps_val = float(data_cfg.get("fps", 24.0))

    # EMA 归一化器
    normalizer_ema: dict[str, Any] = {
        "value": torch.tensor(1.0, device=accelerator.device),
        "initialized": False,
    }

    # 验证用 VideoDecoder
    video_decoder = None
    if valid_every > 0:
        video_decoder = VideoDecoder(cfg["model"]["model_path"], torch.bfloat16, accelerator.device)

    rng = random.Random(cfg.get("seed", 42) + accelerator.process_index)
    noise_generator = torch.Generator(device=accelerator.device).manual_seed(
        cfg.get("seed", 42) + accelerator.process_index
    )

    if accelerator.is_main_process:
        print(f"  退化: 教师={'启用' if teacher_deg else '关闭'}, "
              f"学生={'启用' if student_deg else '关闭'} (mode={deg_mode})")
        print(f"  教师 sigma schedule: {teacher_sigma_schedule}")
        print(f"  学生 sigma0: {sigma0}")
        print(f"  velocity_loss_weight={velocity_loss_weight}, x0_loss_weight={x0_loss_weight}")
        print(f"  dmd_loss_weight={dmd_loss_weight}, regression_loss_weight={regression_loss_weight}")
        print(f"  gen_interval={gen_interval}, critic_interval={critic_interval}")
        print(f"  开始训练, 共 {total_steps} 步 ...")

    # ==================== 训练主循环 ====================
    data_iter = iter(dataloader)
    progress = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    latest_logs: dict[str, float] = {}

    for global_step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        gt_video = batch["pixel"].to(accelerator.device, dtype=torch.bfloat16)
        captions = batch["caption"]
        batch_size = gt_video.shape[0]

        # ----------------------------------------------------------------
        # 阶段 A: 在线数据准备（退化 + VAE encode + spatial upsample + 教师去噪）
        # ----------------------------------------------------------------
        with torch.no_grad():
            # 下采样 + 退化
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
                lq_small_t = (build_low_quality_video(lq_small_clean.clone(), deg_mode, rng, noise_generator)
                              if teacher_deg else lq_small_clean)
                lq_small_s = (build_low_quality_video(lq_small_clean.clone(), deg_mode, rng, noise_generator)
                              if student_deg else lq_small_clean)
                lq_latent_teacher = _encode_and_upsample(lq_small_t)
                lq_latent_student = _encode_and_upsample(lq_small_s)

            conditions = encode_prompts(captions, text_encoder, embeddings_processor)
            conditions_dict = {
                "video_prompt_embeds": conditions["video_prompt_embeds"].to(accelerator.device),
                "prompt_attention_mask": conditions["prompt_attention_mask"].to(accelerator.device),
            }
            noise = torch.randn(
                lq_latent_teacher.shape, device=accelerator.device,
                dtype=lq_latent_teacher.dtype, generator=noise_generator,
            )

        # 教师 3 步 Euler 去噪 → teacher_x0
        teacher_x0 = teacher_euler_denoise(
            transformer, transformer, strategy,
            lq_latent=lq_latent_teacher, noise=noise,
            sigma_schedule=teacher_sigma_schedule,
            conditions=conditions_dict, fps=fps_val,
        )

        # ----------------------------------------------------------------
        # 阶段 B: 学生单步预测
        # ----------------------------------------------------------------
        set_active_branch(transformer, "student")

        num_frames = lq_latent_student.shape[2]
        height = lq_latent_student.shape[3]
        width = lq_latent_student.shape[4]
        shape_meta = {"num_frames": num_frames, "height": height, "width": width, "fps": fps_val}

        # 学生输入: x_t = sigma0 * noise + (1 - sigma0) * lq_latent
        x_t = sigma0 * noise + (1 - sigma0) * lq_latent_student

        video_tokens = strategy._video_patchifier.patchify(x_t)
        video_seq_len = video_tokens.shape[1]
        sigmas_student = torch.full((batch_size,), sigma0, device=x_t.device, dtype=x_t.dtype)
        video_timesteps = sigmas_student.view(-1, 1).expand(-1, video_seq_len)
        video_positions = strategy._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps_val, device=x_t.device, dtype=x_t.dtype,
        )
        video_modality = Modality(
            enabled=True, sigma=sigmas_student, latent=video_tokens,
            timesteps=video_timesteps, positions=video_positions,
            context=conditions_dict["video_prompt_embeds"],
            context_mask=conditions_dict["prompt_attention_mask"],
        )

        velocity_pred, _ = transformer(video=video_modality, audio=None, perturbations=None)

        # 计算 student_x0 和 velocity latent
        velocity_latent = strategy._video_patchifier.unpatchify(
            velocity_pred,
            output_shape=VideoLatentShape(
                batch=batch_size, channels=128,
                frames=num_frames, height=height, width=width,
            ),
        )
        student_x0 = x_t - sigma0 * velocity_latent
        student_x0_tokens = strategy._video_patchifier.patchify(student_x0)
        teacher_x0_tokens = strategy._video_patchifier.patchify(teacher_x0.detach())
        noise_tokens = strategy._video_patchifier.patchify(noise)

        # ----------------------------------------------------------------
        # 阶段 C: 蒸馏损失
        # ----------------------------------------------------------------
        velocity_target = noise_tokens - teacher_x0_tokens
        loss_velocity = F.mse_loss(velocity_pred, velocity_target)
        loss_x0 = F.mse_loss(student_x0, teacher_x0.detach())

        loss_total = velocity_loss_weight * loss_velocity + x0_loss_weight * loss_x0

        latest_logs.update({
            "loss_velocity": loss_velocity.item(),
            "loss_x0": loss_x0.item(),
        })

        # ----------------------------------------------------------------
        # 阶段 D: DMD 损失（按 gen_interval 频率更新）
        # ----------------------------------------------------------------
        do_gen_dmd = (global_step % gen_interval) == 0
        if do_gen_dmd and dmd_loss_weight > 0:
            # 冻结 fake/real LoRA 参数（防止梯度通过共享 base 泄漏）
            fake_lora_mgr = get_lora_manager(transformer, "_lora_fake")
            real_lora_mgr = get_lora_manager(transformer, "_lora_real")
            fake_lora_mgr.requires_grad_(False)
            real_lora_mgr.requires_grad_(False)

            loss_dmd, dmd_logs = compute_dmd_loss(
                transformer, strategy=strategy,
                student_x0_tokens=student_x0_tokens,
                conditions=conditions_dict, shape_meta=shape_meta,
                cfg=cfg, normalizer_ema=normalizer_ema,
            )
            loss_total = loss_total + dmd_loss_weight * loss_dmd

            # 可选 DMD 回归正则
            if regression_loss_weight > 0:
                loss_reg = F.mse_loss(student_x0_tokens.float(), teacher_x0_tokens.float().detach())
                loss_total = loss_total + regression_loss_weight * loss_reg
                latest_logs["loss_dmd_reg"] = loss_reg.item()

            # 恢复 fake/real LoRA 可训练性
            fake_lora_mgr.requires_grad_(True)
            real_lora_mgr.requires_grad_(True)

            latest_logs.update({k: float(v.item()) for k, v in dmd_logs.items()})

        # ----------------------------------------------------------------
        # 阶段 E: GAN 生成器损失
        # ----------------------------------------------------------------
        if gan_enabled and discriminator is not None and do_gen_dmd:
            gan_gen_weight = float(gan_cfg.get("generator_loss_weight", 0.1))
            gan_g_loss, gan_g_logs = compute_gan_generator_loss(discriminator, student_x0_tokens)
            loss_total = loss_total + gan_gen_weight * gan_g_loss
            latest_logs.update({k: float(v.item()) for k, v in gan_g_logs.items()})

        # ----------------------------------------------------------------
        # 阶段 F: 像素空间辅助损失
        # ----------------------------------------------------------------
        loss_pixel_l1 = torch.zeros((), device=accelerator.device)
        loss_lpips = torch.zeros((), device=accelerator.device)
        if pixel_loss_enabled and vae_decoder_nn is not None:
            num_decode = int(pixel_loss_cfg.get("num_decode_frames", 8))
            loss_pixel_l1, loss_lpips = compute_pixel_losses(
                vae_decoder_nn, lpips_fn, student_x0, teacher_x0.detach(), num_frames=num_decode,
            )
            px_l1_w = float(pixel_loss_cfg.get("pixel_l1_loss_weight", 1.0))
            px_lpips_w = float(pixel_loss_cfg.get("lpips_loss_weight", 2.0))
            loss_total = loss_total + px_l1_w * loss_pixel_l1 + px_lpips_w * loss_lpips
            latest_logs["loss_pixel_l1"] = loss_pixel_l1.item()
            latest_logs["loss_lpips"] = loss_lpips.item()

        # ----------------------------------------------------------------
        # 阶段 G: 学生/生成器反向传播 + 优化器更新
        # ----------------------------------------------------------------
        student_optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss_total)
        if max_grad_norm > 0:
            student_grad_norm = accelerator.clip_grad_norm_(student_params, max_grad_norm)
        else:
            student_grad_norm = torch.nn.utils.clip_grad_norm_(student_params, float("inf"))
        student_optimizer.step()
        # 交叉清零（防止陈旧梯度在 generator/critic 之间泄漏）
        student_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        if discriminator_optimizer is not None:
            discriminator_optimizer.zero_grad(set_to_none=True)

        latest_logs["loss_total"] = loss_total.item()
        latest_logs["student_grad_norm"] = float(student_grad_norm)

        # ----------------------------------------------------------------
        # 阶段 H: Fake score critic 更新（按 critic_interval 频率）
        # ----------------------------------------------------------------
        do_critic = (global_step % critic_interval) == 0
        if do_critic:
            critic_optimizer.zero_grad(set_to_none=True)
            loss_critic, critic_logs = compute_fake_score_critic_loss(
                transformer, strategy=strategy,
                student_x0_tokens=student_x0_tokens.detach(),
                conditions=conditions_dict, shape_meta=shape_meta, cfg=cfg,
            )
            accelerator.backward(loss_critic)
            if max_grad_norm > 0:
                critic_grad_norm = accelerator.clip_grad_norm_(critic_params, max_grad_norm)
            else:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, float("inf"))
            critic_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            student_optimizer.zero_grad(set_to_none=True)
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            latest_logs.update({k: float(v.item()) for k, v in critic_logs.items()})
            latest_logs["critic_grad_norm"] = float(critic_grad_norm)

        # ----------------------------------------------------------------
        # 阶段 I: GAN 判别器更新（每步都更新，参考 DMD2）
        # ----------------------------------------------------------------
        if gan_enabled and discriminator is not None:
            gan_disc_weight = float(gan_cfg.get("discriminator_loss_weight", 1.0))
            discriminator_optimizer.zero_grad(set_to_none=True)
            gan_d_loss, gan_d_logs = compute_gan_discriminator_loss(
                discriminator, real_tokens=teacher_x0_tokens, fake_tokens=student_x0_tokens.detach(),
            )
            accelerator.backward(gan_disc_weight * gan_d_loss)
            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad(set_to_none=True)
            latest_logs.update({k: float(v.item()) for k, v in gan_d_logs.items()})

        # 恢复 student 为默认激活分支
        set_active_branch(transformer, "student")

        # ----------------------------------------------------------------
        # 日志 + 保存 + 验证
        # ----------------------------------------------------------------
        step_for_logs = global_step + 1

        if accelerator.is_main_process:
            latest_logs["lr_student"] = float(student_optimizer.param_groups[0]["lr"])
            latest_logs["lr_critic"] = float(critic_optimizer.param_groups[0]["lr"])
            if discriminator_optimizer is not None:
                latest_logs["lr_discriminator"] = float(discriminator_optimizer.param_groups[0]["lr"])
            append_metrics_log(output_dir, step_for_logs, latest_logs)
            if enable_swanlab:
                swanlab.log(latest_logs, step=step_for_logs)

        if accelerator.is_main_process and (global_step % log_every == 0 or global_step == total_steps - 1):
            desc_items = [f"{k}={v:.4g}" for k, v in sorted(latest_logs.items()) if k.startswith("loss")]
            progress.set_description(" ".join(desc_items[:5]) or "training")

        if save_every > 0 and step_for_logs % save_every == 0:
            save_path = save_student_lora_checkpoint(
                accelerator, transformer, checkpoint_dir, step_for_logs, save_metadata,
            )
            if save_fake:
                save_fake_lora_checkpoint(
                    accelerator, transformer, checkpoint_dir, step_for_logs, save_metadata,
                )
            if accelerator.is_main_process and save_path is not None:
                print(f"[step {step_for_logs}] 已保存学生 LoRA: {save_path}")

        if valid_every > 0 and video_decoder is not None and step_for_logs % valid_every == 0:
            transformer.eval()
            run_validation(
                accelerator=accelerator, model=transformer, strategy=strategy,
                video_decoder=video_decoder, teacher_x0=teacher_x0,
                lq_latent=lq_latent_student, student_x0=student_x0,
                output_dir=output_dir, step=step_for_logs, fps=fps_val,
            )
            transformer.train()
            set_active_branch(transformer, "student")

    # ==================== 保存最终 checkpoint ====================
    save_path = save_student_lora_checkpoint(
        accelerator, transformer, checkpoint_dir, total_steps, save_metadata,
    )
    if save_fake:
        save_fake_lora_checkpoint(
            accelerator, transformer, checkpoint_dir, total_steps, save_metadata,
        )
    if accelerator.is_main_process and save_path is not None:
        print(f"[最终] 学生 LoRA 已保存: {save_path}")


if __name__ == "__main__":
    main()
