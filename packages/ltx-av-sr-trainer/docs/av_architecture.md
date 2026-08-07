# LTX-2.3 (22B) 模型架构详解

> 基于 ltx-core 源码分析，覆盖 transformer、text encoder、VAE、position embedding、条件注入等全部组件。

---

## 1. 总体结构

```
┌─────────────────────────────────────────────────────────────────┐
│                        LTXModel (22B)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─ Text Pipeline ──────────────────────────────────────────┐    │
│  │  Gemma-3-12B → FeatureExtractorV2 → EmbeddingsProcessor  │    │
│  │  (Block 1)       (Block 2)            (Block 3)           │    │
│  └───────────────────────────────────────────────────────────┘    │
│                         ↓ video_embeds, audio_embeds              │
│                                                                   │
│  ┌─ Video Path ─────────────┐   ┌─ Audio Path ──────────────┐   │
│  │ patchify_proj: 128→4096  │   │ audio_patchify_proj: 128→2048│ │
│  │ [+cond_video_proj (SR)]  │   │ [+cond_audio_proj (SR)]    │   │
│  │ adaln_single (σ embed)   │   │ audio_adaln_single         │   │
│  │ prompt_adaln_single      │   │ audio_prompt_adaln_single  │   │
│  └───────────┬──────────────┘   └───────────┬────────────────┘   │
│              ↓                               ↓                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │        48 × BasicAVTransformerBlock                        │   │
│  └───────────────────────────────────────────────────────────┘   │
│              ↓                               ↓                    │
│  ┌─ Video Output ───────────┐   ┌─ Audio Output ─────────────┐  │
│  │ scale_shift_table [2,4096]│   │ audio_scale_shift_table    │  │
│  │ norm_out (LayerNorm)     │   │ audio_norm_out             │   │
│  │ proj_out: 4096→128       │   │ audio_proj_out: 2048→128   │   │
│  └──────────────────────────┘   └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心维度参数

| 组件 | Video | Audio |
|------|-------|-------|
| 注意力头数 | 32 | 32 |
| 每头维度 (d_head) | 128 | 64 |
| 隐藏层维度 (inner_dim) | 4096 (32×128) | 2048 (32×64) |
| Patchify 输入维度 | 128 (latent channels) | 128 (8ch × 16mel) |
| Text context dim | 4096 | 2048 |
| 总层数 | 48 (index 0-47) | 48 (共享 block) |

---

## 3. BasicAVTransformerBlock 内部结构

每个 block 包含 7 个子模块，按顺序执行：

```
输入: video_x [B, V_seq, 4096], audio_x [B, A_seq, 2048]

─── ① Video Self-Attention ───
│ scale_shift_table[0:3] + σ_video → scale, shift, gate (AdaLN)
│ RMSNorm(vx) * (1+scale) + shift → norm_vx
│ attn1(norm_vx, pe=video_RoPE) * gate → residual add
│
─── ② Audio Self-Attention ───
│ audio_scale_shift_table[0:3] + σ_audio → scale, shift, gate
│ RMSNorm(ax) * (1+scale) + shift → norm_ax
│ audio_attn1(norm_ax, pe=audio_RoPE) * gate → residual add
│
─── ③ Video Text Cross-Attention (with Prompt AdaLN — LTX-2.3 新增) ───
│ prompt_adaln_single(σ_video) → q_shift, q_scale, q_gate
│ prompt_scale_shift_table + prompt_timestep → kv_shift, kv_scale
│ Q = RMSNorm(vx) * (1+q_scale) + q_shift
│ KV = context * (1+kv_scale) + kv_shift
│ attn2(Q, context=KV) * q_gate → residual add
│
─── ④ Audio Text Cross-Attention (with Prompt AdaLN) ───
│ (同 video，用 audio_attn2 + audio_prompt_scale_shift_table)
│
─── ⑤ A→V Cross-Attention (audio informs video) ───
│ scale_shift_table_a2v_ca_video + σ → scale, shift, gate_a2v
│ vx_scaled = RMSNorm(vx) * (1+scale) + shift
│ ax_scaled = RMSNorm(ax) * (1+scale_audio) + shift_audio
│ audio_to_video_attn(Q=vx_scaled, KV=ax_scaled, pe) * gate_a2v
│ → vx += result
│
─── ⑥ V→A Cross-Attention (video informs audio) ───
│ scale_shift_table_a2v_ca_audio[2:4] + σ → scale, shift, gate_v2a
│ ax_scaled = RMSNorm(ax) * (1+scale) + shift
│ vx_scaled = RMSNorm(vx) * (1+scale_video) + shift_video
│ video_to_audio_attn(Q=ax_scaled, KV=vx_scaled, pe) * gate_v2a
│ → ax += result
│
─── ⑦ Video FFN ───
│ scale_shift_table[3:6] + σ → scale, shift, gate
│ ff(RMSNorm(vx) * (1+scale) + shift) * gate → residual add
│
─── ⑧ Audio FFN ───
│ audio_scale_shift_table[3:6] + σ → scale, shift, gate
│ audio_ff(RMSNorm(ax) * (1+scale) + shift) * gate → residual add

输出: video_x [B, V_seq, 4096], audio_x [B, A_seq, 2048]
```

---

## 4. AdaLN 条件化系统

LTX-2.3 有两套 AdaLN，这是相比 LTX-2 的重要升级：

### 4.1 主 AdaLN (`adaln_single`)

用 per-token timestep (σ) 调制每层的 self-attn 和 FFN：

```
σ_per_token [B, seq] → × timestep_scale_multiplier(1000) → MLP → 6个参数:
  [0:3] → self-attn 的 (shift, scale, gate)
  [3:6] → FFN 的 (shift, scale, gate)
```

### 4.2 Prompt AdaLN (`prompt_adaln_single`) — LTX-2.3 新增

用 per-batch σ 调制 text cross-attention：

```
σ_per_batch [B] → MLP → q_shift, q_scale, q_gate (调制 Q)
prompt_scale_shift_table [2, dim] + prompt_timestep → kv_shift, kv_scale (调制 KV)
```

**设计意图**: 让模型根据当前噪声级别动态调整「听从文本的程度」——高噪声时更依赖文本语义引导，低噪声时更依赖已有图像细节。

### 4.3 Cross-Attention Gate 系统

A↔V cross-attention 有独立的 gate AdaLN 控制信息流强度：

```
av_ca_a2v_gate_adaln_single: σ_video → gate_a2v (标量门控 a→v 信息流)
av_ca_v2a_gate_adaln_single: σ_audio → gate_v2a (标量门控 v→a 信息流)
av_ca_video_scale_shift_adaln_single: σ_video → 4个 scale/shift 参数 (a2v和v2a各2个)
av_ca_audio_scale_shift_adaln_single: σ_audio → 4个 scale/shift 参数
```

---

## 5. Position Embedding (RoPE)

### 5.1 Video — 3D RoPE

位置编码 `[B, 3, seq, 2]`，3 个维度分别编码 (时间, 高度, 宽度)：

- 时间维度: `pixel_coords / fps` → 转为秒（物理时间）
- 空间维度: 像素坐标（经 VAE 压缩后反算）
- `positional_embedding_max_pos = [20, 2048, 2048]`
- `positional_embedding_theta = 10000.0`

### 5.2 Audio — 1D RoPE

位置编码 `[B, 1, seq, 2]`，只编码时间维度：

- `audio_positional_embedding_max_pos = [20]`

### 5.3 Cross-Attention RoPE

A↔V cross-attention 使用 1D RoPE，只用时间维度对齐：

```
cross_pe_max_pos = max(video_max_pos[0], audio_max_pos[0]) = 20
```

这使得 audio 和 video token 通过时间轴自然对齐。

---

## 6. 输入预处理 (TransformerArgsPreprocessor)

### 6.1 Video Preprocessor

```python
x = patchify_proj(latent)             # [B, seq, 128] → [B, seq, 4096]
x = x + cond_video_proj(cond_latent)  # SR 条件注入 (additive, 仅 SR 训练时)
timestep = adaln_single(σ * 1000)     # per-token σ → AdaLN embedding
prompt_ts = prompt_adaln_single(σ_batch)  # per-batch σ → Prompt AdaLN (LTX-2.3)
pe = RoPE(positions)                  # 3D rotary embeddings
context = caption_projection(text)    # LTX-2: 在 transformer 内投影; LTX-2.3: 在 feature_extractor 中已完成
```

### 6.2 Audio Preprocessor

同构，维度从 4096 变为 2048。

### 6.3 MultiModal Preprocessor (额外处理)

为 A↔V cross-attention 准备：

```python
cross_pe = RoPE(positions[:, 0:1, :])  # 只取时间维度，1D
cross_scale_shift_timestep = av_ca_scale_shift_adaln(cross_σ)
cross_gate_timestep = av_ca_gate_adaln(cross_σ)
```

---

## 7. Text Encoder Pipeline (LTX-2.3 = V2)

### 7.1 三阶段流水线

```
Block 1 — Gemma-3-12B LLM:
  text → tokenize(max_length=256, padding="max_length", truncation=True)
       → Gemma forward → hidden_states (tuple of per-layer tensors)

Block 2 — FeatureExtractorV2:
  hidden_states → video_aggregate_embed → per-token RMSNorm → video_features [1, 256, dim]
  hidden_states → audio_aggregate_embed → per-token RMSNorm → audio_features [1, 256, dim]
  (V2 特有: video 和 audio 用不同的聚合权重和投影，V1 只有一个共享的)

Block 3 — EmbeddingsProcessor:
  video_features → right-pad reorder → video_connector (1D Conv blocks) → video_embeds [1, ctx, 4096]
  audio_features → right-pad reorder → audio_connector (1D Conv blocks) → audio_embeds [1, ctx, 2048]
  (V2 特有: video 和 audio connector 有不同的维度, V1 共享)
```

### 7.2 LTX-2 vs LTX-2.3 差异

| 组件 | LTX-2 (19B) | LTX-2.3 (22B) |
|------|-------------|----------------|
| Feature extractor | V1: 单个 `aggregate_embed`，audio=video 复制 | V2: 独立 `video/audio_aggregate_embed` + per-token RMSNorm |
| Caption projection | transformer 内部 (`caption_projection`) | feature_extractor 内部 (Block 2 完成) |
| Connectors | video/audio 同维度 | 独立维度 (4096 vs 2048) |
| Prompt AdaLN | 无 (`cross_attention_adaln=False`) | 有 — 用 σ 调制 text cross-attention |
| Vocoder | HiFi-GAN (`Vocoder`, 24kHz) | BigVGAN v2 + BWE (`VocoderWithBWE`, 48kHz) |

### 7.3 训练中的使用方式

- **预处理阶段**: Block 1+2 (`precompute()`)，保存 video/audio features 到磁盘
- **训练每步**: Block 3 (`create_embeddings()`) 在 `_training_step` 中调用
- **在线训练 (train_online.py)**: Block 1+2 每步实时运行，Block 3 在 `_training_step` 中调用
- **v3 固定 prompt**: Block 1+2 启动时运行一次，缓存结果，卸载 Gemma

---

## 8. SR 条件注入机制

### 8.1 同分辨率 SR (cond_proj)

```python
cond_video_proj = Linear(128, 4096, bias=True)  # 与 patchify_proj 同构
cond_audio_proj = Linear(128, 2048, bias=True)

# 近零初始化（保持 pretrained baseline 稳定）
nn.init.normal_(weight, std=1e-6)
nn.init.zeros_(bias)

# 注入方式:
x = patchify_proj(noisy_hq_latent) + cond_video_proj(lq_latent)
```

### 8.2 跨分辨率 SR (CondSRPatchifyProj)

用于 LQ 和 HQ 空间尺寸不同的场景（如 1280×736 → 2560×1472）：

```python
cond_sr_video_proj = CondSRPatchifyProj(
    channel_proj = Linear(128, 4096),          # 通道映射
    spatial_proj = Linear(lq_h*lq_w, hq_h*hq_w),  # 空间映射 (如 920→3680)
)
```

### 8.3 训练策略 (AVRestorationStrategy)

```
每步:
  - 30% 概率 LQ dropout: cond_latent = zeros (纯 T2AV 模式，防止过度依赖 LQ)
  - 70% 概率 LQ condition: cond_latent = lq_latent + randn * noise_level
    - video noise: uniform(condition_noise_min, condition_noise_max)
    - audio noise: uniform(audio_condition_noise_min, audio_condition_noise_max)
```

---

## 9. Flow Matching 训练

### 9.1 噪声采样

```
σ ~ ShiftedLogitNormalTimestepSampler:
  - shift 根据 sequence length 自动计算:
    min_tokens=1024 → shift=0.95
    max_tokens=4096 → shift=2.05
    capped at 13.0
  - 10% uniform fallback 防止分布塌缩
  - percentile stretching 提高 [0,1] 覆盖率
```

### 9.2 噪声混合

```python
noisy = (1 - σ) * clean + σ * noise    # 线性插值
target = noise - clean                   # velocity prediction
```

### 9.3 Loss 计算

```python
video_loss = MSE(video_pred, video_target) × loss_mask  # 排除 conditioning tokens
audio_loss = MSE(audio_pred, audio_target)
total_loss = video_loss + audio_loss_weight × audio_loss  # 默认 weight=8.0
```

### 9.4 Audio-Video σ 耦合

音频和视频使用相同的 σ（`sigmas = timestep_sampler.sample_for(hq_latents)`），确保两个模态在同一噪声水平去噪，强制学习对齐的时间进程。

---

## 10. 推理管线

```
1. σ schedule: LTX2Scheduler → [σ_max, ..., σ_min] (N steps)
2. 初始噪声: GaussianNoiser → pure noise
3. 每步:
   a. CFGGuider: pred = uncond + guidance_scale × (cond - uncond)
   b. STGGuider: 在指定 block 跳过 self-attn (stg_blocks=[29])
   c. EulerDiffusionStep: x_{t-1} = x_t + (σ_{t-1} - σ_t) × pred
4. 解码:
   a. Video: unpatchify → VAE decoder → tiled_decode → pixels [C,F,H,W]
   b. Audio: unpatchify → AudioDecoder → mel → VocoderWithBWE → waveform 48kHz stereo
```

---

## 11. 参数统计（估算）

| 组件 | 参数量 |
|------|--------|
| patchify_proj + output projections | ~1M |
| audio_patchify_proj + output projections | ~0.5M |
| 48 × video self-attn (4096 dim, 32 heads × 128 d_head) | ~48 × 67M ≈ 3.2B |
| 48 × audio self-attn (2048 dim, 32 heads × 64 d_head) | ~48 × 17M ≈ 0.8B |
| 48 × video text cross-attn (4096 dim) | ~48 × 67M ≈ 3.2B |
| 48 × audio text cross-attn (2048 dim) | ~48 × 17M ≈ 0.8B |
| 48 × video FFN (4096→16384→4096) | ~48 × 134M ≈ 6.4B |
| 48 × audio FFN (2048→8192→2048) | ~48 × 34M ≈ 1.6B |
| 48 × A↔V cross-attn (2 directions, 2048 dim) | ~48 × 34M ≈ 1.6B |
| AdaLN 系统 (主+prompt+cross-attn gate) | ~2B |
| Text encoder (Gemma-3-12B, 推理时卸载) | 12B (不计入) |
| **Transformer 总计** | **~22B** |

---

## 12. Latent Space 常量

| 常量 | 值 | 用途 |
|------|------|------|
| Video latent channels | 128 | VAE encoder/decoder, patchifier |
| 空间压缩比 | 32× (H 和 W) | SpatioTemporalScaleFactors.default() |
| 时间压缩比 | 8× | SpatioTemporalScaleFactors.default() |
| 帧数约束 | `frames % 8 == 1` | 1, 9, 17, ..., 89, 97, 121 |
| 分辨率约束 | W, H 均须被 32 整除 | Config validators |
| Audio latent channels | 8 | AudioLatentShape |
| Audio mel bins | 16 | AudioLatentShape |
| Patchified token dim (video) | 128 (`128×1×1×1`) | Transformer in_channels |
| Patchified token dim (audio) | 128 (`8×16`) | Transformer audio_in_channels |

---

## 13. 关键源码文件

```
packages/ltx-core/src/ltx_core/
├── model/transformer/
│   ├── model.py              # LTXModel — 顶层模型，管理所有组件
│   ├── transformer.py        # BasicAVTransformerBlock — 单个 block 的 forward
│   ├── transformer_args.py   # TransformerArgsPreprocessor — 输入预处理
│   ├── modality.py           # Modality dataclass — 模态数据容器
│   ├── adaln.py              # AdaLayerNormSingle — σ 条件化
│   ├── attention.py          # Attention — Q/K/V + RoPE
│   ├── feed_forward.py       # FeedForward — GELU MLP
│   ├── rope.py               # RoPE 位置编码实现
│   └── cond_sr_patchify.py   # CondSRPatchifyProj — 跨分辨率 SR 投影
├── video_vae/
│   └── video_vae.py          # VideoEncoder/Decoder — 32×32×8 压缩
├── audio_vae/
│   ├── audio_vae.py          # AudioEncoder/Decoder
│   └── vocoder.py            # VocoderWithBWE — BigVGAN + BWE (48kHz)
└── text_encoders/gemma/
    ├── feature_extractor.py  # FeatureExtractorV1/V2
    ├── embeddings_processor.py  # EmbeddingsProcessor (connectors)
    └── embeddings_connector.py  # Embeddings1DConnector
```
