# LTX-2.3 音频数据处理 Pipeline 详解

## 总览

```
[AAC/MP4 音轨] → [PCM 波形] → [重采样] → [Mel 频谱图] → [Audio Latent] → [Patchified Tokens] → [Transformer]
                                                                                                        ↓
[Waveform 48kHz] ← [BWE 3x] ← [Vocoder 16kHz] ← [Mel 频谱图] ← [Audio Latent] ← [Unpatchified] ← [Prediction]
```

---

## 第一阶段：AAC 音轨 → PCM 波形

### 文件格式
训练数据为 MP4 容器，内含 AAC 编码的音频轨道（通常 44100Hz stereo）。

### 解码过程 (`_load_audio_pyav`)
```python
container = av.open(mp4_path)          # 打开 MP4 容器
astream = container.streams.audio[0]    # 获取第一条音频流
sr = astream.rate                       # 原始采样率 (44100Hz)
layout = "stereo"                       # 保留立体声

# AAC → PCM: PyAV 内部调用 FFmpeg 解码
resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=sr)
for frame in container.decode(audio=0):
    frame = resampler.resample(frame)   # 重采样到统一格式
    arr = f.to_ndarray()                # numpy int16 array
    frames.append(torch.from_numpy(arr))

waveform = torch.cat(frames, dim=-1).float() / 32768.0  # int16 → float [-1, 1]
```

### 输出

| 属性 | 值 |
|------|------|
| 格式 | `torch.Tensor` float32 |
| Shape | `[2, 213444]` (stereo, ~4.84s) |
| 采样率 | 44100 Hz |
| 值域 | [-1.0, 1.0] |
| 时长 | `213444 / 44100 = 4.84s` (对齐视频 121帧/25fps=4.8s) |

### 关键细节
- **`s16p` 格式**：signed 16-bit planar（每个声道独立数组），PyAV 标准 PCM 格式
- **`/ 32768.0`**：int16 最大值为 32767，除以 32768 归一化到 [-1, 1]
- **保留 stereo**：不做 downmix，AudioEncoder 原生处理双声道
- **时间裁剪**：按视频 clip 起止时间裁剪音频，不足则 zero-pad

---

## 第二阶段：PCM 波形 → Mel 频谱图

### 重采样

```python
# 44100Hz → 16000Hz（AudioEncoder 的目标采样率）
torchaudio.functional.resample(waveform, 44100, 16000)
```

| 步骤 | 前 | 后 |
|------|------|------|
| 采样率 | 44100 Hz | 16000 Hz |
| 样本数 | 213444 | 77440 |
| 时长 | 4.84s | 4.84s (不变) |
| Shape | `[1, 2, 213444]` | `[1, 2, 77440]` |

**为什么降到 16kHz？** 语音信息集中在 8kHz 以下（奈奎斯特频率 = 16000/2 = 8kHz），16kHz 对语音和大多数音效足够。降采样减少计算量 ~7.6x。

### Mel 频谱图变换

```python
MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,          # FFT 窗口大小
    win_length=1024,     # 窗函数长度 = n_fft
    hop_length=160,      # 帧移（stride）
    f_min=0.0,           # 最低频率
    f_max=8000.0,        # 最高频率 = sr/2
    n_mels=64,           # Mel 滤波器组数量
    window_fn=hann_window,
    center=True,
    pad_mode="reflect",
    power=1.0,           # 幅度谱（非功率谱）
    mel_scale="slaney",
    norm="slaney",
)
```

### STFT → Mel 详解

**Step 1: STFT（短时傅里叶变换）**

```
waveform [1, 2, 77440]
    ↓ 加 Hann 窗 (1024 samples)，每次滑动 160 samples
    ↓ FFT → 复数频谱
STFT output [1, 2, 513, 485]
             ↑  ↑   ↑    ↑
             B  C  freq  time
```

- **频率 bins** = n_fft/2 + 1 = 513（0Hz 到 8000Hz）
- **时间帧** = floor(77440 / 160) + 1 = **485**
- 每帧覆盖 1024/16000 = **64ms**，步进 160/16000 = **10ms**

**Step 2: Mel 滤波器组**

```
STFT [513 freq bins]
    ↓ 64 个三角 Mel 滤波器（Slaney 归一化）
    ↓ 加权求和 → 对数压缩
Mel [64 mel bins]
```

- **Mel 尺度**：模拟人耳对频率的非线性感知（低频分辨率高，高频分辨率低）
- **Slaney 归一化**：每个滤波器面积归一化为 1，防止高频滤波器（更宽）主导
- **power=1.0**：取幅度谱（|STFT|），非功率谱（|STFT|²）

**Step 3: 对数压缩**

```python
mel = torch.log(torch.clamp(mel, min=1e-5))
```

- 对数压缩模拟人耳的响度感知（Weber-Fechner 定律）
- `clamp(min=1e-5)` 防止 log(0) = -inf
- 值域从 [0, ∞) 变为 [-11.5, ∞)

**Step 4: 维度转置**

```python
mel = mel.permute(0, 1, 3, 2)  # [B, C, T, mel] → [B, C, mel, T]... 
# 实际: [B, C, mel_bins, time] → [B, C, time, mel_bins]
```

### Mel 频谱图最终输出

| 属性 | 值 |
|------|------|
| Shape | `[1, 2, 485, 64]` |
| 含义 | `[batch, stereo, time_frames, mel_bins]` |
| dtype | bfloat16 |
| 值域 | [-11.5, ~3.0] (log-mel) |
| 时间分辨率 | 10ms/帧 (hop=160 @ 16kHz) |
| 频率分辨率 | 64 个 Mel 频带 (0~8kHz) |

### 频谱图可视化理解

```
frequency ↑  ████░░░░░░░  8000Hz
(mel)     │  ████████░░░
          │  ██████████░
          │  ████████████  低频（语音基频、低音）
          └──────────────→ time (485 帧, 4.84s)
```

每一列是一个 10ms 的频率快照，越亮（值越大）该频率能量越强。

---

## 第三阶段：Mel 频谱图 → Audio Latent

### AudioEncoder 架构

```
mel [1, 2, 485, 64]     ← 输入: stereo log-mel 频谱图
    │
    ▼ conv_in: Conv2d(2→128, k=3, s=1)
    │  [1, 128, 485, 64]
    │
    ▼ down[0]: ResBlocks + Attention + Downsample(stride=2×2)
    │  [1, 128, 243, 32]      ← 时间 2x↓, 频率 2x↓
    │
    ▼ down[1]: ResBlocks + Attention + Downsample(stride=2×2)  
    │  [1, 256, 122, 16]      ← 时间 2x↓, 频率 2x↓
    │
    ▼ down[2]: ResBlocks + Attention (无下采样)
    │  [1, 512, 122, 16]
    │
    ▼ mid: ResBlock + Attention + ResBlock
    │  [1, 512, 122, 16]
    │
    ▼ norm_out: PixelNorm
    │
    ▼ conv_out: Conv2d(512→16, k=3, s=1)
    │  [1, 16, 122, 16]       ← 注意：输出 16 通道
    │
    ▼ reshape: 16ch → 8ch × 2 (mean + logvar)
    │  mean: [1, 8, 122, 16], logvar: [1, 8, 122, 16]
    │
    ▼ reparameterize: z = mean + std * eps (训练时)
    │                  z = mean (推理时)
    │
    ▼ per_channel_statistics.normalize(z)
       latent [1, 8, 122, 16]  ← 最终 audio latent
```

| 参数 | 值 |
|------|------|
| 总参数量 | 21.3M |
| 时间压缩 | 485 → 122 = **~4x**（2次stride=2下采样，有padding导致非精确4x） |
| 频率压缩 | 64 → 16 = **4x** |
| 通道变换 | 2 (stereo) → 8 (latent channels) |
| 波形总压缩 | 77440 samples → 122×16 = 1952 值 ≈ **40x** |

### 关键设计

- **VAE（变分自编码器）**：conv_out 输出 16 通道 → split 成 mean(8ch) + logvar(8ch) → 重参数化采样
- **PerChannelStatistics**：对 latent 的每个通道做标准化（减均值、除标准差），使 latent 分布接近 N(0,1)，方便 flow matching 加噪
- **Conv2d 架构**：把 mel 频谱图当作 2D "图片" 处理（time × mel_bins），用 2D 卷积提取时频特征

---

## 第四阶段：Audio Latent → Patchified Tokens

### AudioPatchifier

```python
# AudioPatchifier(patch_size=1)
latent [1, 8, 122, 16]
    ↓ reshape: (B, C, T, M) → (B, T, C*M)
tokens [1, 122, 128]
       ↑   ↑    ↑
       B  seq  token_dim = 8 × 16 = 128
```

| 变换 | 说明 |
|------|------|
| 输入 | `[B, 8, 122, 16]` — 4D latent |
| 输出 | `[B, 122, 128]` — 2D token 序列 |
| token_dim | 8 × 16 = 128（和视频 token_dim 一致） |
| seq_len | 122（每个时间步一个 token） |

**为什么 token_dim = 128？** 和视频的 patchified token dim 完全一致（视频: 128 latent channels × 1×1×1 patch = 128），使得 transformer 的 patchify_proj / audio_patchify_proj 输入维度统一。

---

## 第五阶段：Tokens → Transformer 内部

### 进入 Transformer

```
audio tokens [B, 122, 128]
    ↓ audio_patchify_proj (Linear 128 → 2048)  ← 注意：audio hidden dim = 2048
audio hidden [B, 122, 2048]

（对比视频: patchify_proj Linear 128 → 4096，hidden dim = 4096）
```

### SR 条件注入

```
LQ audio tokens [B, 122, 128]
    ↓ cond_audio_proj (Linear 128 → 128)
    ↓ ADD 到 noisy audio tokens (在 patchify_proj 之前)
conditioned tokens [B, 122, 128]
    ↓ audio_patchify_proj
    → 进入 transformer blocks
```

---

## 解码 Pipeline（推理时）

### AudioDecoder

```
latent [1, 8, 122, 16]          ← transformer 预测 → unpatchify
    │
    ▼ per_channel_statistics.un_normalize(z)
    │
    ▼ conv_in: Conv2d(8→512)
    ▼ mid blocks
    ▼ up[0]: ResBlocks (无上采样)
    ▼ up[1]: ResBlocks + Upsample(2x)  ← 时间 2x↑, 频率 2x↑
    ▼ up[2]: ResBlocks + Upsample(2x)  ← 时间 2x↑, 频率 2x↑
    ▼ conv_out: Conv2d(128→2)
    │
mel [1, 2, 485, 64]             ← 还原 log-mel 频谱图
```

| 参数 | 值 |
|------|------|
| 总参数量 | 31.9M |
| 时间恢复 | 122 → 485 ≈ 4x↑ |
| 频率恢复 | 16 → 64 = 4x↑ |
| 通道恢复 | 8 → 2 (stereo) |

### Vocoder（BigVGAN）

```
mel [1, 2, 485, 64]
    ↓ permute → [1, 2, 64, 485]    ← (B, C, mel_bins, time)
    ↓ 对每个声道独立处理
    ↓ ConvTranspose1d 上采样链
    ↓ ...
wav_16k [1, 2, 77600]              ← 16kHz waveform
    ↓ 
    ↓ duration = 77600/16000 = 4.85s
```

| 属性 | 值 |
|------|------|
| 输出采样率 | 16000 Hz |
| Mel→Wav 上采样 | 485 time frames → 77600 samples = 160x (= hop_length) |

### BWE（Bandwidth Extension，带宽扩展）

```
wav_16k [1, 2, 77600] @ 16kHz
    ↓ 插值 3x 上采样
    ↓ BWE 残差网络预测高频分量
    ↓ wav + residual
wav_48k [1, 2, 232800] @ 48kHz
```

| 属性 | 值 |
|------|------|
| 输入 | 16kHz waveform |
| 输出 | 48kHz waveform |
| 上采样倍率 | 3x |
| 恢复频段 | 8kHz~24kHz（人耳可听上限附近的高频细节）|
| 时长 | 232800/48000 = 4.85s (不变) |

---

## 全链路数据维度总表

| 阶段 | Shape | 数据类型 | 大小(元素数) |
|------|-------|---------|------------|
| AAC 音轨 | 压缩文件 | — | ~50KB |
| PCM 波形 44kHz | `[2, 213444]` | float32 | 426,888 |
| PCM 波形 16kHz | `[2, 77440]` | float32 | 154,880 |
| Log-Mel 频谱 | `[2, 485, 64]` | bfloat16 | 62,080 |
| **Audio Latent** | **`[8, 122, 16]`** | **bfloat16** | **15,616** |
| Patchified tokens | `[122, 128]` | bfloat16 | 15,616 |
| Transformer hidden | `[122, 2048]` | bfloat16 | 249,856 |

**总压缩率：426,888 → 15,616 ≈ 27x**

---

## 与视频数据对比

| 属性 | 视频 | 音频 |
|------|------|------|
| 原始输入 | 1920×1152 pixels × 121 frames | stereo × 213444 samples |
| 原始元素数 | 798M | 427K |
| Latent shape | `[128, 16, 36, 60]` | `[8, 122, 16]` |
| Latent 元素数 | 4,423,680 | 15,616 |
| Patchified tokens | 34,560 | 122 |
| Token dim | 128 | 128 |
| Hidden dim | 4096 | 2048 |
| Prompt embed dim | 4096 | 2048 |
| 压缩率 | ~180x | ~27x |
| Token 占比 | 99.6% | 0.4% |

---

## 音频 VAE 的数学原理

### 编码（VAE Encoder）
```
给定 mel 频谱 x:
    encoder 输出 μ(x), log σ²(x)     ← mean 和 log-variance
    z = μ + σ · ε,  ε ~ N(0, I)      ← 重参数化技巧
    z_norm = (z - μ_dataset) / σ_dataset  ← 数据集级标准化
```

### 解码（VAE Decoder）
```
给定 latent z_norm:
    z = z_norm · σ_dataset + μ_dataset    ← 反标准化
    mel_hat = decoder(z)                   ← 还原 mel 频谱
```

### Flow Matching 训练目标
```
给定 clean latent x₀ 和 noise ε ~ N(0, I):
    σ ~ ShiftedLogitNormal               ← 和视频共享同一个 σ
    x_t = (1-σ)·x₀ + σ·ε                ← 加噪
    target = ε - x₀                      ← velocity prediction
    loss = MSE(model(x_t), target)
```
