# Pixel + Waveform 无损条件注入方案设计文档

## 1. 动机与目标

### 当前方案的瓶颈

现有 latent-space 条件注入方案（`sft_2x_latent.yaml`）的训练流程：

```
Raw MP4 → HQ pixel (1280×768) → VAE encode → HQ latent [B, 128, 16, 24, 40]  (target)
        → LQ pixel (640×384)  → VAE encode → LQ latent [B, 128, 15, 12, 20]  (condition)
                                                ↓
                                    CondLatent2xProj → tokens → additive inject
```

**问题**：
1. 每个训练 step 需要 2 次 Video VAE tiled encode + 2 次 Audio VAE encode
2. VAE encode 是主要的计算和显存瓶颈（22B transformer forward 反而可以用 gradient checkpointing 压下来）
3. 推理时 LQ 输入也需要过一次 VAE encode，增加延迟
4. VAE encode 本身是有损的——重建误差在条件注入时传播到生成结果

### 新方案目标

- **训练时**：完全消除 LQ condition 的 VAE encode，仅保留 HQ target 的 VAE encode
- **推理时**：LQ 条件直接以 pixel/waveform 输入，无需 VAE encode
- **零信息损失**：所有空间/时间信息通过 reshape（space-to-depth）进入 channel 维，无 pooling/stride/interpolation
- **保持注入接口不变**：仍然是 additive inject 到 patchify 后的 hidden space

---

## 2. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          训练数据流                                          │
│                                                                             │
│  Raw MP4 ──→ HQ pixel (1280×768×121F) ──→ VAE encode ──→ HQ latent (target)│
│         │                                                                   │
│         └─→ LQ pixel (640×384×121F) ──→ PixelCondProj ──→ additive inject   │
│                                                                             │
│  Raw Audio ─→ LQ waveform [2, T] ──→ WaveCondProj ──→ additive inject      │
│                                                                             │
│  [无 VAE encode for LQ condition]                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 与现有方案对比

| 维度 | 现有 Latent 方案 | 新 Pixel+Wave 方案 |
|------|-----------------|-------------------|
| 训练每 step VAE 开销 | 2×Video + 2×Audio encode | 仅 1×Video encode (HQ target) |
| 推理 VAE 开销 | LQ encode + HQ decode | 仅 HQ decode |
| LQ 条件信息保真度 | 有损 (VAE 重建误差) | 近无损 (CausalConv 特征提取) |
| Projector 参数量 | ~2.88M | ~28.3M |
| 预期训练速度提升 | baseline | ~2-3x (省去 LQ VAE encode) |
| 注入方式 | additive | additive (不变) |

---

## 3. Video: PixelCondProj 详细设计

### 3.1 核心思想：FlashVSR 风格 Causal Pixel Projection

参考 FlashVSR 的 `Buffer_LQ4x_Proj` 架构，使用 PixelShuffle（space-to-depth）+ CausalConv3d 实现时间压缩，
通过首帧 repeat padding 天然对齐 VAE 的 causal temporal 行为。

**与纯 reshape (space-to-depth + MLP) 方案的对比**：

| | 纯 reshape + MLP | CausalConv 方案 (FlashVSR 风格) |
|---|---|---|
| 首帧处理 | 必须丢末帧 (121%8≠0) | 首帧 repeat padding, 无信息丢失 |
| 时间交互 | 无 (每 patch 独立) | CausalConv 提供跨帧感受野 |
| VAE 对齐 | 有 1 帧固定偏移 | 精确对齐 (同为 causal 架构) |
| 特征质量 | MLP 从扁平 pixel 学特征 | Conv 提取局部时空特征 |

### 3.2 维度推导

```
LQ pixel: [B, 3, 121, 384, 640]
目标: 输出 token count = HQ latent tokens = 16 × 24 × 40 = 15,360

VAE temporal 公式: latent_T = (pixel_T - 1) / 8 + 1 = (121-1)/8 + 1 = 16
VAE spatial 公式: latent_H = pixel_H / 32 = 768/32 = 24, latent_W = 1280/32 = 40
LQ spatial / 16: 384/16 = 24, 640/16 = 40  ← 与 HQ latent spatial 精确匹配

Step 1: 首帧 repeat padding 7 帧
  [B, 3, 121+7, 384, 640] = [B, 3, 128, 384, 640]

Step 2: PixelShuffle3d(1, 16, 16) — 空间 space-to-depth
  [B, 3×1×16×16, 128, 384/16, 640/16] = [B, 768, 128, 24, 40]

Step 3: CausalConv3d × 3 (stride_t=2 each) — 8x 时间压缩
  128 → 64 → 32 → 16 temporal frames ← 精确匹配 VAE latent_T = 16 ✓

Step 4: flatten + Linear proj
  [B, 1024, 16, 24, 40] → [B, 16×24×40, 1024] → Linear → [B, 15360, 4096]
```

### 3.3 首帧 Repeat Padding — 与 VAE 的 Causal 对齐

**VAE 的 causal 行为**：
```
输入 121 帧: [f0, f1, ..., f120]
causal conv3d: f0 作为 causal seed
输出 (121-1)/8 + 1 = 16 temporal tokens
  token 0 编码 [f0 ~ f8]
  token k 编码 [f(8k+1) ~ f(8k+8)]
```

**PixelCondProj 的 causal 行为**（参考 FlashVSR）：
```
Padding: [f0,f0,f0,f0,f0,f0,f0, f0,f1,...,f120] = 128 帧
  ↓ PixelShuffle + CausalConv3d ×3 (stride_t=2 each)
  = 16 temporal outputs
  ↓ 去掉 output[0] (纯 padding 的 causal seed)
  = 16 temporal tokens

对齐:
  output[0]:  编码 [f0 附近区域] ← 对齐 VAE token 0 (编码 f0 ~ f8)
  output[1]:  编码 [f1 ~ f8 附近] ← 对齐 VAE token 1
  ...
  output[15]: 编码 [f113 ~ f120] ← 对齐 VAE token 15
```

**无需丢弃任何帧，121 帧全部参与编码，输出 16 temporal tokens 精确匹配 VAE。**

### 3.4 模块架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  PixelCondProj (adapted from FlashVSR for LTX-2.3)                 │
│  ═══════════════════════════════════════════════════                 │
│                                                                     │
│  输入: [B, 3, 121, 384, 640]  (LQ pixel, [-1, 1])                  │
│                                                                     │
│  Step 1: First-frame repeat padding (7 帧)                          │
│  ─────────────────────────────────────────────                      │
│    cat([f0×7, video]) → [B, 3, 128, 384, 640]                      │
│                                                                     │
│  Step 2: PixelShuffle3d(1, 16, 16) — 空间 space-to-depth            │
│  ─────────────────────────────────────────────                      │
│    [B, 3, 128, 384, 640] → [B, 768, 128, 24, 40]                   │
│                                                                     │
│  Step 3: CausalConv3d stack — 8x 时间压缩                           │
│  ─────────────────────────────────────────────                      │
│    conv1: CausalConv3d(768→512, k=(4,3,3), s=(2,1,1)) + RMS + SiLU │
│           128 → 64 temporal, 空间 3×3 局部交互                       │
│    conv2: CausalConv3d(512→768, k=(4,1,1), s=(2,1,1)) + RMS + SiLU │
│           64 → 32 temporal                                          │
│    conv3: CausalConv3d(768→1024, k=(4,1,1), s=(2,1,1)) + RMS + SiLU│
│           32 → 16 temporal                                          │
│                                                                     │
│  Step 4: Flatten + Linear projection                                 │
│  ─────────────────────────────────────────────                      │
│    rearrange → [B, 15360, 1024]                                     │
│    Linear(1024, 4096) → [B, 15360, 4096]                            │
│                                                                     │
│  输出: [B, 15360, 4096]  ← 与 HQ latent tokens 精确对齐             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 参考实现

见 `src/echo_sr/model/pixel_cond_proj.py`。

关键设计决策：
- **Layer 1 使用 (4,3,3) kernel**：PixelShuffle 后相邻空间位置是相邻 16×16 patch，
  3×3 conv 捕获跨 patch 边界的连续性特征
- **Layer 2-3 使用 (4,1,1) kernel**：仅做时间压缩，空间交互交给 transformer self-attention，
  大幅降低参数量 (从 200M+ 降至 23M)
- **near_zero init**：仅对最后的 Linear proj 做近零初始化，conv 层保持默认初始化

### 3.6 参数量明细

| 子模块 | 结构 | 参数量 |
|--------|------|--------|
| conv1 | CausalConv3d(768→512, k=4×3×3) | 14,156,288 |
| norm1 + conv2 | RMSNorm(512) + CausalConv3d(512→768, k=4×1×1) | 1,574,148 |
| norm2 + conv3 | RMSNorm(768) + CausalConv3d(768→1024, k=4×1×1) | 3,147,520 |
| norm3 + proj | RMSNorm(1024) + Linear(1024→4096) | 4,199,420 |
| **Video 总计** | | **~23.08M** |

### 3.7 分辨率泛化性

由于使用卷积（非固定维度的 Linear），模块天然支持变分辨率：
- 时间: 任何 T (8n+1 格式)，padding 7帧 → (T+7)/8 - 1 = (T-1)/8 temporal tokens
- 空间: 任何 H×W (只要 H%16==0, W%16==0)

测试验证通过:
```
[121, 384, 640]  → [15360, 4096]  ✓
[121, 768, 1280] → [61440, 4096]  ✓  (1K 分辨率)
[81, 256, 448]   → [4928, 4096]   ✓  (低分辨率)
[33, 128, 256]   → [640, 4096]    ✓  (极短视频)
```

---

## 4. Audio: WaveCondProj 详细设计

### 4.1 为什么需要 STFT 变换

#### 信号处理分析

Raw waveform 中，频率信息是全局分布在时域的——一个 440Hz 正弦波的每一个 sample 都长得一样（+/-交替），单独看一小段 chunk 无法判断其频率。

而预训练的 audio transformer 接收的是 mel-spectrogram 经 VAE 压缩后的 latent：
```
Audio VAE 处理链:
  Waveform → STFT → Mel Filter Bank → Log Scale → VAE Encode → Latent [8, T, 16]
```

Transformer 的内部表征已经"习惯"了频域特征（谐波关联、频带能量分布等）。

#### STFT 是否"有损"

**STFT 是严格无损的线性变换**：
- 满足 Parseval 定理：`||x||² = ||STFT(x)||²`
- 给定 complex spectrogram（real + imaginary）+ 窗函数参数，ISTFT 可精确重建原波形
- 本质是正交基变换（离散时间分段 DFT），与 mel scale（不可逆）、magnitude-only（丢相位）等有损操作有本质区别

#### STFT vs Mel-Spectrogram 的关系

```
Raw Waveform
    ↓  STFT (线性可逆变换)
Linear Spectrogram [freq_bins × time]    ← 频率轴线性均匀 (0, Δf, 2Δf, ...)
    ↓  Mel Filter Bank (矩阵乘法，不可逆 — 多对一映射)
Mel Spectrogram [mel_bins × time]        ← 频率轴按人耳感知非线性压缩
    ↓  Audio VAE Encoder (非线性，有损)
Audio Latent [8, frames, 16]             ← 模型实际使用的
```

STFT spectrogram 是 mel-spectrogram 的**上游无损表示**：
- 信息量: STFT ⊃ Mel-Spec ⊃ Audio Latent
- STFT 保留了完整的相位信息 + 线性频率分辨率
- Mel-Spec 是 STFT 的一个有损线性投影（mel filter bank 是多对一映射，不可逆）

使用 STFT 作为 condition 输入的优势：
1. 信息严格包含 mel-spec 的全部信息，projector 理论上可学到 mel-spec 能提供的一切
2. 额外保留了相位信息，有助于时间精确对齐
3. 时频表示对小 MLP 友好——频率特征显式暴露，无需隐式学 DFT

#### 为什么不用 raw waveform 直接 chunk

| 对比 | Raw Waveform Chunk | STFT Spectrogram |
|------|-------------------|-----------------|
| 每 token 语义 | 640 个连续采样点（高频震荡数值） | 一个时间帧的完整频谱（641 个频率 bin） |
| MLP 需学的映射 | 先隐式学 DFT → 再学特征 | 直接学频率加权（类似 mel filter） |
| 类比 | "把莫尔斯电码翻译成法语" | "把英语翻译成法语"（同语系） |
| 信息密度 | 低（时域高度冗余） | 高（时频解耦） |

### 4.2 维度推导

目标：输出 token count 必须匹配 audio patchifier 的 token count = **121**。

```
Audio pipeline (LTX-2.3):
  Waveform (44.1kHz, 4.84s) → resample to 16kHz → 77,440 samples
  Audio VAE: STFT → mel → encode → latent [B, 8, 121, 16]
  AudioPatchifier: [B, 8, 121, 16] → "b c t f -> b t (c f)" → [B, 121, 128]
  audio_patchify_proj: Linear(128, 2048) → [B, 121, 2048]

我们的目标: Waveform → ... → [B, 121, 2048]
```

关键参数选择：
```
输入: 16kHz × 4.84s = 77,440 samples
选择 hop_length = 640:
  STFT time frames = 77440 / 640 = 121 精确！ (无需 padding/interpolation)

选择 n_fft = 1280:
  freq_bins = n_fft / 2 + 1 = 641

STFT 输出: [B, 2_channels, 641_freq, 121_time] (complex)
展开 real+imag: [B, 4, 641, 121]  (4 = stereo × real/imag)

按时间帧 patchify (每帧一个 token，频率轴全部展平):
  [B, 4, 641, 121] → rearrange "b c f t -> b t (c f)" → [B, 121, 2564]

Linear(2564, 2048) → [B, 121, 2048]  ← 精确对齐 audio transformer tokens
```

### 4.3 模块架构

```
┌─────────────────────────────────────────────────────────────────┐
│  WaveCondProj                                                   │
│  ═══════════════════════════════════════════════════             │
│                                                                 │
│  输入: [B, 2, T_samples]  (stereo waveform, [-1, 1])            │
│                                                                 │
│  Step 1: Resample to 16kHz (匹配模型时间轴)                      │
│  ─────────────────────────────────────────────                  │
│    torchaudio.functional.resample(44100 → 16000)                │
│    [B, 2, ~213k] → [B, 2, 77440]                               │
│    (线性插值重采样，近无损)                                       │
│                                                                 │
│  Step 2: STFT (完全可逆的线性变换)                               │
│  ─────────────────────────────────────────────                  │
│    n_fft=1280, hop_length=640, window=hann                      │
│    [B, 2, 77440] → [B, 2, 641, 121] (complex)                  │
│    展开 real+imag: → [B, 4, 641, 121]                           │
│    可逆性: ISTFT(STFT(x)) = x (精确重建)                        │
│                                                                 │
│  Step 3: 按时间帧 Patchify (纯 reshape，无损)                    │
│  ─────────────────────────────────────────────                  │
│    rearrange: "b c f t -> b t (c f)"                            │
│    [B, 4, 641, 121] → [B, 121, 2564]                           │
│    每个 token = 一个时间帧的完整频谱 (641 freq × 4 channels)     │
│                                                                 │
│  Step 4: MLP Projection                                         │
│  ─────────────────────────────────────────────                  │
│    Linear(2564, 2048)                                           │
│    [B, 121, 2564] → [B, 121, 2048]                             │
│                                                                 │
│  输出: [B, 121, 2048]  ← 与 audio transformer tokens 精确对齐   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 参考实现

```python
import torch
import torch.nn as nn
from einops import rearrange


class WaveCondProj(nn.Module):
    """Lossless waveform condition projector via STFT + patchify + MLP.

    Processing chain:
      1. Resample to target_sr (16kHz, matching model's temporal alignment)
      2. STFT: invertible linear transform → complex spectrogram
      3. Patchify: reshape time frames into token sequence (zero info loss)
      4. MLP: project to audio transformer hidden dim

    STFT is a strictly lossless linear transform (Parseval's theorem):
    given complex output (real + imag), ISTFT exactly reconstructs the input.

    hop_length=640 is chosen so that STFT produces exactly target_tokens=121
    time frames from 16kHz × 4.84s = 77440 samples, requiring no interpolation.

    Args:
        audio_inner_dim: Audio transformer hidden dim (2048 for LTX-2.3).
        target_tokens: Required output token count (121, matching AudioPatchifier).
        n_fft: STFT window size (1280 → 641 freq bins).
        hop_length: STFT hop length (640 → exactly 121 time frames).
        source_sr: Input waveform sample rate.
        target_sr: Model's expected sample rate for temporal alignment.
        n_channels: Input audio channels (2 for stereo).
    """

    def __init__(
        self,
        audio_inner_dim: int = 2048,
        target_tokens: int = 121,
        n_fft: int = 1280,
        hop_length: int = 640,
        source_sr: int = 44100,
        target_sr: int = 16000,
        n_channels: int = 2,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.target_tokens = target_tokens
        self.n_channels = n_channels

        freq_bins = n_fft // 2 + 1  # 641
        # 4 = n_channels (stereo) × 2 (real + imag)
        patch_dim = n_channels * 2 * freq_bins  # 2 × 2 × 641 = 2564

        # Register STFT window as buffer (not a parameter)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, audio_inner_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 2, T_samples] stereo waveform in [-1, 1].

        Returns:
            [B, target_tokens, audio_inner_dim] = [B, 121, 2048]
        """
        B, C, T = waveform.shape

        # Step 1: Resample to target_sr
        if self.source_sr != self.target_sr:
            import torchaudio.functional as AF
            waveform = AF.resample(waveform, self.source_sr, self.target_sr)
        # [B, 2, 77440]

        # Pad/trim to ensure exactly target_tokens frames from STFT
        expected_samples = (self.target_tokens) * self.hop_length  # 121 × 640 = 77440
        if waveform.shape[-1] > expected_samples:
            waveform = waveform[..., :expected_samples]
        elif waveform.shape[-1] < expected_samples:
            pad_len = expected_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Step 2: STFT (lossless linear transform)
        # Process all channels together: [B*C, T] → [B*C, freq, time] complex
        x = waveform.reshape(B * C, -1)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=False,  # no padding → exact frame count
        )
        # spec: [B*C, 641, 121] complex

        # Reshape back to batch: [B, C, 641, 121]
        spec = spec.reshape(B, C, spec.shape[1], spec.shape[2])

        # Split complex into real + imag: [B, C, 641, 121] → [B, 2C, 641, 121]
        spec = torch.cat([spec.real, spec.imag], dim=1)
        # [B, 4, 641, 121]

        # Step 3: Patchify by time frame (pure reshape, zero info loss)
        tokens = rearrange(spec, "b c f t -> b t (c f)")
        # [B, 121, 2564]

        # Step 4: Project to audio hidden dim
        tokens = self.proj(tokens)
        # [B, 121, 2048]
        return tokens

    @staticmethod
    def init_near_zero(module: "WaveCondProj", std: float = 1e-6) -> None:
        """Near-zero init for stable training start."""
        last_linear = module.proj[-1] if isinstance(module.proj, nn.Sequential) else module.proj
        nn.init.normal_(last_linear.weight, std=std)
        nn.init.zeros_(last_linear.bias)
```

### 4.5 参数量明细

| 子模块 | 结构 | 参数量 |
|--------|------|--------|
| Linear(2564, 2048) | 2564×2048 + 2048 | 5,253,120 |
| **Audio 总计** | | **~5.25M** |

注：如果需要更强的表达能力，可扩展为多层 MLP：
```python
# 增强版 (可选)
self.proj = nn.Sequential(
    nn.Linear(2564, 1024),   # 2.63M
    nn.SiLU(),
    nn.Linear(1024, 2048),   # 2.10M
)
# 总计: ~4.73M (略少但多了非线性)
```

### 4.6 STFT 参数选择理由

| 参数 | 值 | 理由 |
|------|---|------|
| n_fft | 1280 | freq_bins = 641，足够覆盖 8kHz 内所有频率 |
| hop_length | 640 | 77440 / 640 = 121 精确对齐 audio token count，无需 interpolation |
| window | Hann | 标准选择，频率泄漏小 |
| center | False | 不做 padding，保证精确帧数 = samples / hop |
| return_complex | True | 保留完整相位信息（real + imag），确保 STFT 可逆 |

### 4.7 关于重采样 (44.1kHz → 16kHz)

重采样是本方案中唯一的"近似"操作：
- `torchaudio.functional.resample` 使用 sinc 插值（Kaiser window），在奈奎斯特频率内近似无损
- 8kHz 以上的信号会被 anti-aliasing filter 截断，但模型的 audio path 本身也工作在 16kHz
- 如果追求极致无损，可以在 44.1kHz 上直接做 STFT（调整 hop 使 time_frames=121），但 freq_bins 会更大（增加参数量）

---

## 5. 完整数据流图

```
                    ┌───────────────────────────────────────────────────┐
                    │              Training Step                         │
                    └───────────────────────────────────────────────────┘

  ┌─────────────────┐                           ┌────────────────────┐
  │ HQ Video Pixel  │                           │ LQ Video Pixel     │
  │[B,3,121,768,1280]                           │[B,3,121,384,640]   │
  └────────┬────────┘                           └────────┬───────────┘
           │                                             │
           ▼                                             ▼
  ┌─────────────────┐                           ┌────────────────────┐
  │  Video VAE Enc  │                           │  PixelCondProj     │
  │  (tiled encode) │                           │  (CausalConv stack │
  │                 │                           │   + proj, ~23M)    │
  └────────┬────────┘                           └────────┬───────────┘
           │                                             │
           ▼                                             ▼
  [B, 128, 16, 24, 40]                         [B, 15360, 4096]
           │                                             │
           ▼                                             │
  ┌─────────────────┐                                    │
  │ video_patchify  │                                    │
  │ → [B,15360,128] │                                    │
  │ → patchify_proj │                                    │
  │ → [B,15360,4096]│                                    │
  └────────┬────────┘                                    │
           │                                             │
           ▼                                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    x = hq_tokens + cond_tokens                │
  │                    (additive injection)                       │
  └──────────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     48× BasicAVBlock (22B)     │
                    │     + Audio Cross-Attention    │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │       Velocity Prediction      │
                    │      [B, 15360, 128]           │
                    └────────────────────────────────┘


  ┌─────────────────┐                           ┌────────────────────┐
  │ HQ Audio Waveform│                          │ LQ Audio Waveform  │
  │ [B, 2, ~213k]   │                           │ [B, 2, ~213k]      │
  └────────┬────────┘                           └────────┬───────────┘
           │                                             │
           ▼                                             ▼
  ┌─────────────────┐                           ┌────────────────────┐
  │  Audio VAE Enc  │                           │   WaveCondProj     │
  │                 │                           │  (STFT + patchify  │
  │                 │                           │   + MLP, ~5.25M)   │
  └────────┬────────┘                           └────────┬───────────┘
           │                                             │
           ▼                                             ▼
  [B, 8, 121, 16]                               [B, 121, 2048]
           │                                             │
           ▼                                             │
  ┌─────────────────┐                                    │
  │ audio_patchify  │                                    │
  │ → [B, 121, 128] │                                    │
  │ → patchify_proj │                                    │
  │ → [B, 121, 2048]│                                    │
  └────────┬────────┘                                    │
           │                                             │
           ▼                                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              audio_x = hq_tokens + cond_tokens               │
  │                    (additive injection)                       │
  └──────────────────────────────────────────────────────────────┘
```

---

## 6. 训练策略适配

### 6.1 修改后的 _train_step

```python
def _train_step(self, batch: dict) -> torch.Tensor:
    """Training step: 仅 HQ VAE encode, LQ 直接 pixel/wave 注入."""
    device = self.device

    # HQ: 仍需 VAE encode (target 在 latent space)
    hq_video = batch["hq_video"].to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        hq_latent = self.vae_encoder.tiled_encode(hq_video)

    # LQ: 直接传 pixel，由 PixelCondProj 处理（无需 VAE encode！）
    lq_video_pixel = batch["lq_video"].to(device=device, dtype=torch.bfloat16)

    # Audio: LQ 直接传 waveform，由 WaveCondProj 处理
    hq_audio = batch["hq_audio"].to(device=device)
    lq_audio_waveform = batch["lq_audio"].to(device=device)
    with torch.no_grad():
        hq_audio_latent = self._encode_audio(hq_audio, ...)

    # Assemble batch — LQ 现在是 pixel/waveform 而非 latent
    assembled = {
        "hq_latents": {"latents": hq_latent, ...},
        "lq_pixel": lq_video_pixel,        # NEW: pixel tensor
        "lq_waveform": lq_audio_waveform,   # NEW: waveform tensor
        ...
    }
    # Strategy forward...
```

### 6.2 Condition Dropout 适配

```python
# 在 strategy 中:
if drop_cond:
    video_cond_tokens = torch.zeros(B, 15360, 4096, device=device, dtype=dtype)
else:
    video_cond_tokens = self.pixel_cond_proj(lq_video_pixel)
    if condition_noise_max > 0:
        # Noise 直接加在 token space (而非 pixel space)
        noise_level = rng.uniform(noise_min, noise_max)
        video_cond_tokens = video_cond_tokens + torch.randn_like(video_cond_tokens) * noise_level
```

### 6.3 Condition Noise 策略变化

原方案在 latent space 加 noise。新方案有两个选择：

1. **在 pixel/waveform 空间加 noise**（模拟真实 LQ 退化）
2. **在 projected token 空间加 noise**（类似原方案语义）

建议：**在 token 空间加 noise**（方案 2），因为：
- 与原方案行为一致，训练策略可复用
- Pixel 空间加 noise 可能导致不自然的退化模式
- Token space 的 noise 直接影响 transformer 输入，正则化效果更直接

---

## 7. 推理流程

```
推理 (无 LQ VAE encode):

LQ Video Pixel [1, 3, 121, 384, 640]
    ↓ PixelCondProj
    → [1, 15360, 4096]  (condition tokens)

LQ Audio Waveform [1, 2, 213444]
    ↓ WaveCondProj
    → [1, 121, 2048]  (condition tokens)

Random Noise [1, 128, 16, 24, 40]
    ↓ patchify_proj
    → [1, 15360, 4096]
    + video condition tokens  (additive)
    ↓ Denoise (N steps)
    ↓ unpatchify
    → [1, 128, 16, 24, 40]
    ↓ VAE Decode
    → HQ Video [1, 3, 121, 768, 1280]
```

**推理省去的计算**：1 次 Video VAE encode + 1 次 Audio VAE encode。

---

## 8. 总参数量预算

| 模块 | 参数量 |
|------|--------|
| PixelCondProj (Video) | 23,077,376 (~23.1M) |
| WaveCondProj (Audio) | 5,253,120 (~5.25M) |
| **Cond Proj 总计** | **~28.3M** |
| LoRA (rank=384) | ~100M |
| **总可训练参数** | **~128M** |

对比原方案 cond_proj 仅 2.88M，新方案增加约 8M 参数，但省去了每 step 两次 VAE encode 的计算。这是**参数换计算**的合理 trade-off。

---

## 9. 实现计划

### Phase 1: 核心模块
- [ ] 实现 `PixelCondProj` 模块
- [ ] 实现 `WaveCondProj` 模块
- [ ] 单元测试：验证输出维度、near-zero 初始化

### Phase 2: 集成
- [ ] 修改 `loader.py`：支持新的 `cond_proj.type = "pixel_wave"`
- [ ] 修改 `strategy.py`：接收 pixel/waveform 而非 latent 作为 LQ condition
- [ ] 修改 `trainer.py`：训练 step 中跳过 LQ 的 VAE encode
- [ ] 新增 config: `configs/sft_2x_pixel_wave.yaml`

### Phase 3: 验证
- [ ] 对比训练速度（预期 2-3x 提升）
- [ ] 对比显存占用（预期显著降低）
- [ ] 生成质量对比（与 latent 方案 A/B test）

### Phase 4: 推理适配
- [ ] 修改 `validator.py`：推理时直接传 pixel/waveform
- [ ] 修改 `infer.py`：移除 LQ VAE encode 步骤

---

## 10. 风险与缓解

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| MLP 表达力不足，无法从 6144-dim patch 中提取有效特征 | 中 | 增大 mid_dim (512→1024)，或加 conv 层 |
| Pixel space 与 latent space 语义 gap 导致注入效果差 | 中 | Near-zero init + 更高的 condition_noise 适配期 |
| STFT 重采样引入极微量信息损失 | 低 | 可选择在原始 44.1kHz 做 STFT (调整 hop) |
| 训练初期 loss 比 latent 方案更高 | 低 | 预期行为，projector 需要更多 step 收敛 |

---

## 11. 参考

- Space-to-Depth: 最早在 PixelShuffle (Shi et al., 2016) 的逆操作中使用
- Vision Transformer patchify: ViT (Dosovitskiy et al., 2020) 使用相同的 reshape 策略
- STFT 可逆性: Griffin & Lim (1984), 任何保留 complex 值的 STFT 天然可逆
- 当前 LQ Proj 设计: `echo-sr-trainer/docs/LQ_PROJ_REPORT.md`
- 当前 CondLatent2xProj: `echo-sr-trainer/src/echo_sr/model/cond_latent_2x.py`
