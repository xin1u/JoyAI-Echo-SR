# Echo-SR

[中文说明](README_zh.md)

Echo-SR is a DMD-only research release for one-step stage-2 video
super-resolution on LTX models. It includes two independently reproducible
recipes:

| Recipe | Base model | Data path | DMD design | Launcher |
| --- | --- | --- | --- | --- |
| LTX-2 19B DMD | LTX-2 19B dev | WebDataset videos | Online 3-step teacher + student/real/fake LoRA + GAN | `scripts/train_dmd_19b.sh` |
| LTX-2.3 22B DMD | LTX-2.3 22B dev | WebDataset videos | Online 3-step teacher + student/real/fake LoRA + GAN | `scripts/train_dmd_22b.sh` |

Both public configs require `dmd.enabled: true`. Each trainer checks this flag
before loading model weights and aborts if DMD is disabled. Plain non-DMD
distillation trainers are intentionally not part of this release.

The code is video-only. Audio parameters are not trained, and the provided
inference entry does not emit an audio track.

## Repository Layout

```text
configs/echo_sr_ltx2_19b_dmd.yaml       19B online DMD recipe
configs/echo_sr_ltx23_22b_dmd.yaml      22B online WebDataset DMD recipe
configs/accelerate/fsdp_8gpu.yaml       single-node FSDP configuration
packages/ltx-core/                      compatible LTX core snapshot
packages/ltx-pipelines/                 compatible LTX pipelines snapshot
packages/ltx-trainer/                   compatible LTX trainer utilities
packages/ltx-sr-trainer/                Echo-SR DMD datasets and trainers
scripts/train_dmd_19b.sh                19B launcher
scripts/train_dmd_22b.sh                22B launcher
scripts/infer.sh                        one-video generator inference
```

Datasets, checkpoints, logs, and generated media are excluded from Git.

## Requirements

- Linux and NVIDIA GPUs
- Python 3.11 or 3.12
- PyTorch 2.7 or newer and a matching CUDA environment
- Eight H200-class GPUs are recommended for the public FSDP settings
- `ffmpeg` and `ffprobe` available on `PATH`

Install with `uv`:

```bash
git clone git@github.com:xin1u/Echo-SR.git
cd Echo-SR
uv sync --all-packages --all-extras
source .venv/bin/activate
```

Or install into an existing CUDA environment:

```bash
python -m pip install -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer \
  -e 'packages/ltx-sr-trainer[tracking]'
```

## Checkpoints

The latest Echo-SR generator weights are released at
<https://huggingface.co/xin1u/Echo-SR>:

```bash
hf download xin1u/Echo-SR --local-dir checkpoints/echo-sr
```

The two model families require different base checkpoints, official distilled
LoRAs, and spatial upsamplers. Do not mix 19B and 22B assets. Public YAML files
use this layout:

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

LTX-2.3 base assets are published by Lightricks at
<https://huggingface.co/Lightricks/LTX-2.3>. Use the generator filename from the
Echo-SR model page as `--student-lora-path` or `--lora-path`. Record the code
commit, model revision, filename, and checksum for every experiment.

## WebDataset Training Data

Both recipes use the same WebDataset pipeline and decode videos online from tar shards. Each sample
contains a matching `.mp4` and `.json`; metadata must include `height`, `width`,
and at least one configured English caption path. See
`examples/sample_metadata.example.json`.

Generate the shard index:

```bash
python tools/build_train_index.py \
  'data/shards/*.tar' \
  --output data/train_index.json
```

The public configs train at `1024x1536`, `121` frames, and `24 fps`. Each builds
the stage-2 condition online, runs the frozen three-step teacher, and optimizes
the one-step student with distillation, DMD, GAN, L1, and LPIPS losses.

## Validate Configs

Validation does not load checkpoints or initialize CUDA:

```bash
python packages/ltx-sr-trainer/scripts/train_stage2_sr_dmd_webdataset.py \
  configs/echo_sr_ltx2_19b_dmd.yaml --validate-config

python packages/ltx-sr-trainer/scripts/train_stage2_sr_dmd_webdataset.py \
  configs/echo_sr_ltx23_22b_dmd.yaml --validate-config
```

Successful output explicitly reports that DMD is enabled.

## Training

Single-node eight-GPU 19B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29531 \
bash scripts/train_dmd_19b.sh
```

Single-node eight-GPU 22B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29532 \
bash scripts/train_dmd_22b.sh
```

Override configuration or environment without editing launchers:

```bash
CONFIG=configs/echo_sr_ltx23_22b_dmd.yaml \
ECHO_SR_VENV=/path/to/venv \
NPROC_PER_NODE=8 \
bash scripts/train_dmd_22b.sh
```

For multi-node runs, set the same `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and
`NPROC_PER_NODE` on every node, and assign a distinct `NODE_RANK`.

SwanLab is optional. Set `swanlab.enable_swanlab: true`, install the tracking
extra, and inject the key only at runtime:

```bash
export SWANLAB_API_KEY='...'
```

No API key is read from YAML. Outputs include `run_config.yaml`, scalar logs,
validation comparisons, and generator checkpoints under
`<output_dir>/checkpoints/`.

## Inference

Run either DMD generator on one low-resolution video by selecting matching
family assets:

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

Frame count must satisfy `8*k+1`; target height and width must be divisible by
64.

## License and Attribution

This repository contains a modified compatibility snapshot of LTX code. See
`NOTICE.md` for provenance and modifications. The repository is distributed
under the included LTX-2 Community License Agreement. Model assets may carry
additional terms from their release pages.
