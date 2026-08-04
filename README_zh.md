# Echo-SR

[English](README.md)

Echo-SR 是面向 LTX 系列模型的二阶段视频超分训练与推理代码。本仓库公开了两套实际使用的
一步 SR 训练方案，并附带与其兼容的 LTX 源码快照，目标是让使用者拿到仓库后能够直接准备
权重、数据并复现实验。

当前代码只训练和推理视频分支。LTX 的音频分支保持冻结，推理脚本也不会输出音轨。

## 两套训练方案

| 方案 | 基座 | 训练目标 | 可训练模块 | 启动脚本 |
| --- | --- | --- | --- | --- |
| Echo-SR LTX-2.3 22B | LTX-2.3 22B dev | 三步教师在线蒸馏到一步学生 | 视频 LoRA | `scripts/train_22b.sh` |
| Echo-SR LTX-2 19B DMD | LTX-2 19B dev | 在线蒸馏 + DMD + token-pooling GAN | 学生 LoRA、fake-score LoRA、判别器 | `scripts/train_19b_dmd.sh` |

示例配置统一使用 `1024x1536`、`121` 帧、`24 fps` 和 2 倍 latent 空间上采样器。
教师 sigma 为 `[0.909375, 0.725, 0.421875, 0.0]`，蒸馏后的学生只执行
`0.909375 -> 0.0` 一次去噪。

> 历史内部脚本名容易产生歧义：`run_distill.sh` 对应 22B，
> `run_distill_dmd_v2.sh` 对应 19B DMD。本仓库已用明确的公开入口重新命名。

## 目录结构

```text
configs/                         两套训练配置和 FSDP 配置
examples/                        WebDataset 元数据示例
packages/ltx-core/               兼容的 LTX core 源码
packages/ltx-pipelines/          兼容的 LTX pipelines 源码
packages/ltx-trainer/            兼容的 LTX trainer 工具
packages/ltx-sr-trainer/         Echo-SR 数据、损失和训练入口
scripts/train_22b.sh             22B 训练入口
scripts/train_19b_dmd.sh         19B DMD/GAN 训练入口
scripts/infer.sh                 单视频教师/学生推理入口
tools/build_train_index.py       根据 tar shard 生成索引
```

模型权重、数据、日志和推理视频均不进入 Git。

## 环境

- Linux + NVIDIA GPU
- Python 3.11 或 3.12
- CUDA 12.8 同等级环境
- PyTorch 2.7 或更高版本
- 示例 FSDP 配置建议使用 8 张 H200 同等级 GPU；实际显存取决于分辨率、帧数、LoRA rank
  和启用的损失项
- 系统可直接调用 `ffmpeg`

使用 `uv` 安装：

```bash
git clone git@github.com:xin1u/Echo-SR.git
cd Echo-SR
uv sync --all-packages --all-extras
source .venv/bin/activate
```

也可以安装到已有 CUDA 环境：

```bash
python -m pip install -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer \
  -e 'packages/ltx-sr-trainer[tracking]'
```

## 准备模型

下载匹配的基座、蒸馏 LoRA、空间上采样器、Gemma 文本编码器和 LPIPS VGG 权重，放入
`checkpoints/`，或修改 YAML 中对应路径。

LTX-2.3 官方文件位于 <https://huggingface.co/Lightricks/LTX-2.3>。22B 示例配置默认目录：

```text
checkpoints/
├── gemma-3-12b/
├── lpips_vgg.pth
├── ltx-2.3-22b-dev.safetensors
├── ltx-2.3-22b-distilled-lora-384.safetensors
└── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

19B 方案应使用 `configs/echo_sr_ltx2_19b_dmd.yaml` 中列出的 LTX-2 19B 对应文件，不能把
22B 和 19B 的基座、LoRA 或 upsampler 混用。

Echo-SR 训练权重不会直接提交到 Git，最新权重发布在
<https://huggingface.co/xin1u/Echo-SR>。可直接下载到独立目录：

```bash
hf download xin1u/Echo-SR --local-dir checkpoints/echo-sr
```

推理时将模型页中的实际权重文件传给 `--student-lora-path`。实验记录应同时保存代码
commit、模型 revision、文件名和 SHA256，保证后续可追溯。

## 准备数据

训练数据采用流式 WebDataset tar shard。每个样本由同名 `.mp4` 与 `.json` 组成：

```text
000001.mp4
000001.json
000002.mp4
000002.json
```

JSON 至少要包含 `height`、`width` 和配置所需的一个 caption 字段；完整格式见
`examples/sample_metadata.example.json`。可选时序字段包括 `fps`、`fps_target`、
`frame_num_set`、`high_quality_frame_index`。

根据 tar shard 生成 `data/train_index.json`：

```bash
python tools/build_train_index.py \
  'data/shards/*.tar' \
  --output data/train_index.json
```

数据加载时先保持长宽比缩放，再中心裁剪到目标尺寸。长视频会随机采样；帧数低于目标但不低于
`min_frames` 的视频会镜像补帧。公开配置的单卡 batch size 为 1。

## 配置检查

下面的命令只检查 YAML，不加载模型和 CUDA：

```bash
python packages/ltx-sr-trainer/scripts/train_stage2_sr_distill.py \
  configs/echo_sr_ltx23_22b.yaml --validate-config

python packages/ltx-sr-trainer/scripts/train_stage2_sr_distill_dmd_v2.py \
  configs/echo_sr_ltx2_19b_dmd.yaml --validate-config
```

## 训练

单节点 8 卡训练 22B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29531 \
bash scripts/train_22b.sh
```

单节点 8 卡训练 19B DMD/GAN：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29532 \
bash scripts/train_19b_dmd.sh
```

可通过环境变量覆盖配置和虚拟环境：

```bash
CONFIG=configs/echo_sr_ltx23_22b.yaml \
ECHO_SR_VENV=/path/to/venv \
NPROC_PER_NODE=8 \
bash scripts/train_22b.sh
```

多机时，每个节点设置相同的 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、
`NPROC_PER_NODE`，并分别设置 `NODE_RANK=0...NNODES-1`。

启用 SwanLab 时，安装 `tracking` extra，将 YAML 中 `enable_swanlab` 改为 `true`，并在
运行时注入密钥：

```bash
export SWANLAB_API_KEY='...'
```

代码不会从 YAML 读取密钥。训练权重保存在 `<output_dir>/checkpoints/`，验证对比视频保存在
`<output_dir>/validation/`，实际运行配置会复制到 `<output_dir>/run_config.yaml`。

## 推理

使用一步学生 LoRA 对一个低分辨率视频进行超分：

```bash
bash scripts/infer.sh \
  --input-video examples/input_lq.mp4 \
  --prompt 'A detailed cinematic scene.' \
  --output-dir outputs/inference/demo \
  --checkpoint-path checkpoints/ltx-2.3-22b-dev.safetensors \
  --student-lora-path outputs/echo_sr_ltx23_22b/checkpoints/lora_weights_step_30000.safetensors \
  --spatial-upsampler-path checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root checkpoints/gemma-3-12b \
  --target-height 1024 \
  --target-width 1536 \
  --num-frames 121 \
  --fps 24
```

增加 `--teacher-lora-path` 可以同时输出三步教师结果。帧数必须满足 `8*k+1`，目标高宽必须
能被 64 整除。当前推理入口只生成无音轨视频。

## 复现要点

- 22B 的教师和学生都从官方蒸馏 LoRA 初始化，教师冻结并以三步生成监督，学生执行一步去噪。
- 只优化视频 LoRA，音频 LoRA 参数冻结。
- 像素域监督在抽样帧上计算 L1 与 LPIPS。
- 19B 方案额外训练 fake-score LoRA 和轻量 token-pooling 判别器。
- 启动脚本会先进入仓库根目录，因此 YAML 中相对路径均相对于仓库根目录解析。

## 许可证与来源

仓库包含经过修改的 LTX 源码快照，来源和修改说明见 `NOTICE.md`。整体按仓库内附带的
LTX-2 Community License Agreement 发布；模型文件还需遵守各自发布页面上的附加条款。

Echo-SR 是与 JoyAI-Echo 生态相关的研究实现，但不代表上游项目对本仓库作出官方背书。
