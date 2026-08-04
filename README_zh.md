# Echo-SR

[English](README.md)

Echo-SR 是面向 LTX 系列模型的一步二阶段视频超分 DMD 训练与推理代码。本仓库只公开
DMD 方案，包含两个可以独立复现的版本：

| 方案 | 基座 | 数据形式 | DMD 结构 | 启动脚本 |
| --- | --- | --- | --- | --- |
| LTX-2 19B DMD | LTX-2 19B dev | WebDataset 视频 | 在线三步教师 + student/real/fake LoRA + GAN | `scripts/train_dmd_19b.sh` |
| LTX-2.3 22B DMD | LTX-2.3 22B dev | WebDataset 视频 | 在线三步教师 + student/real/fake LoRA + GAN | `scripts/train_dmd_22b.sh` |

两份公开配置都必须设置 `dmd.enabled: true`。trainer 会在加载权重前检查该字段，关闭
DMD 会直接退出。普通非 DMD 蒸馏 trainer 不在本次开源范围内。

当前代码只训练视频分支，音频参数不参与训练，推理入口也不生成音轨。

## 目录结构

```text
configs/echo_sr_ltx2_19b_dmd.yaml       19B 在线 WebDataset DMD 配置
configs/echo_sr_ltx23_22b_dmd.yaml      22B 在线 WebDataset DMD 配置
configs/accelerate/fsdp_8gpu.yaml       单节点 FSDP 配置
packages/ltx-core/                      兼容的 LTX core 源码
packages/ltx-pipelines/                 兼容的 LTX pipelines 源码
packages/ltx-trainer/                   兼容的 LTX trainer 工具
packages/ltx-sr-trainer/                Echo-SR DMD 数据与训练代码
scripts/train_dmd_19b.sh                19B 启动入口
scripts/train_dmd_22b.sh                22B 启动入口
scripts/infer.sh                        单视频生成器推理
```

数据、权重、日志和生成视频不会提交到 Git。

## 环境安装

- Linux + NVIDIA GPU
- Python 3.11 或 3.12
- PyTorch 2.7 或更高版本及匹配的 CUDA
- 公开 FSDP 配置建议使用单节点 8 张 H200 同等级 GPU
- 系统可调用 `ffmpeg` 和 `ffprobe`

使用 `uv`：

```bash
git clone git@github.com:xin1u/Echo-SR.git
cd Echo-SR
uv sync --all-packages --all-extras
source .venv/bin/activate
```

安装到已有 CUDA 环境：

```bash
python -m pip install -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer \
  -e 'packages/ltx-sr-trainer[tracking]'
```

## 权重

最新 Echo-SR 生成器权重发布在 <https://huggingface.co/xin1u/Echo-SR>：

```bash
hf download xin1u/Echo-SR --local-dir checkpoints/echo-sr
```

19B 和 22B 必须分别使用匹配的基座、官方蒸馏 LoRA 和空间上采样器，不能跨模型混用。
公开 YAML 默认使用下面的目录：

```text
checkpoints/
├── gemma-3-12b/
├── lpips_vgg.pth
├── ltx-2-19b-dev.safetensors
├── ltx-2-19b-distilled-lora-384.safetensors
├── ltx-2-spatial-upscaler-x2-1.0.safetensors
├── ltx-2.3-22b-dev.safetensors
├── ltx-2.3-22b-distilled-lora-384.safetensors
└── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

LTX-2.3 官方资产位于 <https://huggingface.co/Lightricks/LTX-2.3>。推理时，将 Echo-SR
模型页对应生成器文件传给 `--student-lora-path` 或 `--lora-path`。实验记录应同时保存代码
commit、模型 revision、文件名和校验值。

## WebDataset 训练数据

19B 和 22B 使用完全相同的 WebDataset 加载链路，从 tar shard 在线解码视频。每个样本需要同名 `.mp4` 和 `.json`；
JSON 至少包含 `height`、`width` 和配置指定的一个英文 caption。格式见
`examples/sample_metadata.example.json`。

生成 shard 索引：

```bash
python tools/build_train_index.py \
  'data/shards/*.tar' \
  --output data/train_index.json
```

两份公开配置均为 `1024x1536`、`121` 帧、`24 fps`。训练阶段在线构造 stage-2 条件，运行冻结的
三步教师，并使用蒸馏、DMD、GAN、L1 和 LPIPS 训练一步学生。

## 配置检查

下面的命令不会加载模型或初始化 CUDA：

```bash
python packages/ltx-sr-trainer/scripts/train_stage2_sr_dmd_webdataset.py \
  configs/echo_sr_ltx2_19b_dmd.yaml --validate-config

python packages/ltx-sr-trainer/scripts/train_stage2_sr_dmd_webdataset.py \
  configs/echo_sr_ltx23_22b_dmd.yaml --validate-config
```

成功输出会明确显示 DMD 已开启。

## 训练

单节点 8 卡训练 19B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29531 \
bash scripts/train_dmd_19b.sh
```

单节点 8 卡训练 22B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29532 \
bash scripts/train_dmd_22b.sh
```

可通过环境变量覆盖配置和虚拟环境：

```bash
CONFIG=configs/echo_sr_ltx23_22b_dmd.yaml \
ECHO_SR_VENV=/path/to/venv \
NPROC_PER_NODE=8 \
bash scripts/train_dmd_22b.sh
```

多机时，每个节点设置相同的 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、
`NPROC_PER_NODE`，并使用不同的 `NODE_RANK`。

SwanLab 为可选项。将 YAML 中 `enable_swanlab` 改为 `true`，安装 tracking extra，并仅在
运行时传入密钥：

```bash
export SWANLAB_API_KEY='...'
```

代码不会从 YAML 读取密钥。输出包括 `run_config.yaml`、标量日志、验证对比和
`<output_dir>/checkpoints/` 下的生成器权重。

## 推理

选择匹配的模型资产，可对 19B 或 22B DMD 生成器执行单视频推理：

```bash
bash scripts/infer.sh \
  --input-video input_lq.mp4 \
  --prompt 'A detailed cinematic scene.' \
  --output-dir outputs/inference \
  --checkpoint-path checkpoints/ltx-2.3-22b-dev.safetensors \
  --student-lora-path checkpoints/echo-sr/GENERATOR.safetensors \
  --spatial-upsampler-path checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root checkpoints/gemma-3-12b \
  --target-height 1024 \
  --target-width 1536 \
  --num-frames 121 \
  --fps 24
```

帧数必须满足 `8*k+1`，目标高宽必须能被 64 整除。

## 许可证与来源

仓库包含修改后的 LTX 兼容源码快照，来源和修改内容见 `NOTICE.md`。整体按仓库附带的
LTX-2 Community License Agreement 发布，模型资产还需遵守各自发布页面上的附加条款。
