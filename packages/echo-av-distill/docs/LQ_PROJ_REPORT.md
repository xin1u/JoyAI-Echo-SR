# LQ Proj 详细报告

## 概述

LQ Proj（Low-Quality Projection）是 LTX-2.3 超分辨率训练框架中的关键模块，负责将低质量(LQ)条件 latent 映射到 Transformer 的隐空间，以加性方式注入，实现条件化去噪。

**核心设计理念**：
- 跨分辨率 SR：LQ 1280×736 → HQ 1920×1152（latent: 40×23 → 60×36）
- 学习型空间映射（非简单插值）
- 近零初始化，保证训练稳定性

---

## 模型架构

### 1. CondSRPatchifyProj（视频分支）

```
输入: [B, 128, T, 23, 40]  -- LQ latent (5D)
输出: [B, T×36×60, 4096]   -- HQ token space
```

**三阶段处理：**

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Pre-Conv Refinement (per-frame, LQ resolution)        │
│  ───────────────────────────────────────────────────────────    │
│  [B,128,T,23,40] → reshape → [B×T, 128, 23, 40]                │
│                                    ↓                            │
│              Conv2d(128,128,3×3) + SiLU + Conv2d(128,128,3×3)   │
│                                    ↓                            │
│                          + residual connection                  │
│                                    ↓                            │
│                        [B×T, 128, 23, 40]                       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Learned Spatial Mapping                               │
│  ───────────────────────────────────────────────────────────    │
│  [B×T, 128, 23×40] = [B×T, 128, 920]                           │
│                          ↓                                      │
│            Linear(920, 2160)  -- position mapping               │
│                          ↓                                      │
│  [B×T, 128, 2160] = [B×T, 128, 36×60]                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Channel Projection                                    │
│  ───────────────────────────────────────────────────────────    │
│  reshape → [B, T×2160, 128]                                     │
│                   ↓                                             │
│         Linear(128, 4096)  -- to transformer hidden dim         │
│                   ↓                                             │
│           [B, T×2160, 4096]                                     │
└─────────────────────────────────────────────────────────────────┘
```

**参数量明细：**

| 子模块 | 结构 | 参数量 |
|--------|------|--------|
| pre_conv | 2× Conv2d(128,128,3×3) | 295,168 |
| spatial_proj | Linear(920, 2160) | 1,989,360 |
| proj | Linear(128, 4096) | 528,384 |
| **视频总计** | | **2,812,912** |

### 2. cond_audio_proj（音频分支）

```
输入: [B, T_audio, 32]   -- patchified LQ audio latent
输出: [B, T_audio, 2048] -- audio hidden dim
```

**结构：**
```
Linear(32, 2048)  -- 简单线性映射
```

| 子模块 | 结构 | 参数量 |
|--------|------|--------|
| cond_audio_proj | Linear(32, 2048) | 67,584 |
| **音频总计** | | **67,584** |

### 3. 总参数量

| 模块 | 参数量 |
|------|--------|
| CondSRPatchifyProj (Video) | 2,812,912 |
| cond_audio_proj (Audio) | 67,584 |
| **总计** | **2,880,496 (~2.88M)** |

---

## 初始化策略

### 近零初始化 (Near-Identity Init)

确保训练开始时 SR 条件几乎不影响输出，模型从预训练的 T2AV baseline 稳定启动。

```python
@staticmethod
def init_near_identity(module: CondSRPatchifyProj, proj_std: float = 1e-6):
    # 1. spatial_proj: 最近邻映射初始化
    with torch.no_grad():
        weight = torch.zeros(hq_h * hq_w, lq_h * lq_w)
        for hq_idx in range(hq_h * hq_w):
            hq_y, hq_x = hq_idx // hq_w, hq_idx % hq_w
            lq_y = min(int(hq_y * lq_h / hq_h), lq_h - 1)
            lq_x = min(int(hq_x * lq_w / hq_w), lq_w - 1)
            lq_idx = lq_y * lq_w + lq_x
            weight[hq_idx, lq_idx] = 1.0
        module.spatial_proj.weight.copy_(weight)
        module.spatial_proj.bias.zero_()
    
    # 2. proj: 近零初始化
    nn.init.normal_(module.proj.weight, std=1e-6)
    nn.init.zeros_(module.proj.bias)
```

**效果：**
- `spatial_proj` 初始行为 ≈ 最近邻上采样
- `proj` 输出 ≈ 0，SR 条件初始贡献为零
- `pre_conv` 默认初始化 + 残差连接 ≈ identity

---

## 注入方式

### 加性注入 (Additive Injection)

在 `TransformerArgsPreprocessor.prepare()` 中：

```python
def prepare(self, modality: Modality, ...) -> TransformerArgs:
    # patchify_proj: 将 HQ noisy latent 映射到 hidden space
    x = self.patchify_proj(modality.latent)  # [B, T×H×W, 4096]
    
    # 加性注入 LQ condition
    if self.cond_proj is not None and modality.cond_latent is not None:
        x = x + self.cond_proj(modality.cond_latent)  # 直接相加
    
    # ... 后续处理 (timestep, RoPE, etc.)
    return TransformerArgs(x=x, ...)
```

**关键点：**
- 条件通过 **element-wise add** 注入（不是 concat/cross-attn）
- 注入发生在 **patchify 之后、Transformer blocks 之前**
- LQ condition 和 HQ noisy latent 在同一 token 位置对齐

---

## 训练策略

### 1. Condition Dropout (10%)

```yaml
training_strategy:
  lq_drop_prob: 0.1
```

```python
drop_cond = self._rng.random() < self.config.lq_drop_prob
if drop_cond:
    video_cond_latent = torch.zeros_like(lq_latents)  # 全零替换
else:
    video_cond_latent = lq_latents
```

**目的**：保留 T2AV 生成能力，避免过度依赖 LQ condition。

### 2. Condition Noise Injection (0.4-0.6)

```yaml
training_strategy:
  condition_noise_min: 0.4
  condition_noise_max: 0.6
  audio_condition_noise_min: 0.4
  audio_condition_noise_max: 0.6
```

```python
if self.config.condition_noise_max > 0:
    noise_level = self._rng.uniform(
        self.config.condition_noise_min, 
        self.config.condition_noise_max
    )
    cond_noise = torch.randn_like(video_cond_latent) * noise_level
    video_cond_latent = video_cond_latent + cond_noise
```

**目的**：增强鲁棒性，模拟真实 LQ 输入的噪声/压缩失真。

### 3. Cross-Resolution 处理

```python
# Strategy 中的跨分辨率检测
cross_res_sr = (lq_latents.shape[-2:] != hq_latents.shape[-2:])

if cross_res_sr:
    # 保持 5D 直接传给 CondSRPatchifyProj
    video_cond_latent = lq_latents  # [B, 128, T, 23, 40]
else:
    # 同分辨率：先 patchify 再用简单 Linear
    video_cond_latent = patchify(lq_latents)  # [B, T×H×W, 128]
```

---

## 数据流图

```
                    ┌──────────────────┐
                    │  LQ Video Latent │
                    │ [B,128,T,23,40]  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   10% dropout?   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │ YES          │              │ NO
              ▼              │              ▼
    ┌─────────────────┐      │    ┌───────────────────┐
    │ zeros_like(lq)  │      │    │ + noise (0.4-0.6) │
    └────────┬────────┘      │    └─────────┬─────────┘
             │               │              │
             └───────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │ CondSRPatchifyProj│
                    │   (2.81M params)  │
                    └────────┬─────────┘
                             │
                    [B, T×36×60, 4096]
                             │
                    ┌────────▼─────────┐
                    │   Additive Add   │
                    │  x = hq + cond   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ 48× BasicAVBlock │
                    │  (22B params)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Velocity Pred  │
                    │ [B, T×36×60, 128]│
                    └──────────────────┘
```

---

## Checkpoint 格式

### 保存时

```python
# trainer._save_checkpoint()
for k, v in full_sd.items():
    if "_cond_" in k and "proj" in k:
        state[f"diffusion_model.{k}"] = v.detach().cpu()
```

**保存的 key 列表：**
```
diffusion_model._cond_sr_video_proj.pre_conv.0.weight
diffusion_model._cond_sr_video_proj.pre_conv.0.bias
diffusion_model._cond_sr_video_proj.pre_conv.2.weight
diffusion_model._cond_sr_video_proj.pre_conv.2.bias
diffusion_model._cond_sr_video_proj.spatial_proj.weight
diffusion_model._cond_sr_video_proj.spatial_proj.bias
diffusion_model._cond_sr_video_proj.proj.weight
diffusion_model._cond_sr_video_proj.proj.bias
diffusion_model._cond_audio_proj.weight
diffusion_model._cond_audio_proj.bias
```

### 与 LoRA 合并保存

单个 `.safetensors` 同时包含：
- LoRA A/B weights（~100M params）
- cond_proj weights（~2.88M params）

---

## 配置示例

```yaml
# configs/sft_1k.yaml
data:
  hq_resolution: [1920, 1152, 121]  # HQ pixel: W×H×F
  lq_resolution: [1280, 736, 121]   # LQ pixel: W×H×F
  # latent: HQ 60×36, LQ 40×23 (pixel / 32)

training_strategy:
  lq_drop_prob: 0.1                 # 10% condition dropout
  condition_noise_min: 0.4          # noise range for video
  condition_noise_max: 0.6
  audio_condition_noise_min: 0.4    # noise range for audio
  audio_condition_noise_max: 0.6
```

---

## 与同分辨率 SR 的对比

| 方面 | 同分辨率 (init_cond_proj) | 跨分辨率 (init_cond_sr_proj) |
|------|---------------------------|------------------------------|
| 模块 | Linear(128, 4096) | CondSRPatchifyProj |
| 输入 | Patchified 3D | Raw 5D latent |
| 空间映射 | 无（相同大小） | 学习型 Linear(920, 2160) |
| 参数量 | ~0.5M | ~2.8M |
| 使用场景 | 去噪/去模糊 | 超分辨率 |

---

## 关键实现位置

| 功能 | 文件位置 |
|------|----------|
| CondSRPatchifyProj 定义 | `ltx-core/.../transformer/cond_sr_patchify.py` |
| init_cond_sr_proj | `ltx-core/.../transformer/model.py:366` |
| cond_proj 注入 | `ltx-core/.../transformer/transformer_args.py:157-158` |
| init_cond_proj (echo) | `echo-sr-trainer/.../model/loader.py:67` |
| 训练时 LQ 处理 | `echo-sr-trainer/.../training/strategy.py:156-190` |
| 推理时 LQ 处理 | `echo-sr-trainer/.../validation/validator.py:157-163` |
