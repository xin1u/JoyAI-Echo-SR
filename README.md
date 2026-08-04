# Echo-SR

[中文说明](README_zh.md)

Echo-SR is a research training and inference release for one-step stage-2 video
super-resolution on LTX models. It contains the two recipes used in our
experiments and the compatible LTX source packages needed to reproduce them.

This repository is video-only. The audio branch of LTX is frozen and is not
trained or emitted by the provided inference script.

## Recipes

| Public recipe | Base model | Training objective | Trainable parts | Launcher |
| --- | --- | --- | --- | --- |
| Echo-SR LTX-2.3 22B | LTX-2.3 22B dev | 3-step teacher to 1-step student online distillation | Video LoRA | `scripts/train_22b.sh` |
| Echo-SR LTX-2 19B DMD | LTX-2 19B dev | Online distillation + DMD + token-pooling GAN | Student LoRA, fake-score LoRA, discriminator | `scripts/train_19b_dmd.sh` |

Both example configs train at `1024x1536`, `121` frames, and `24 fps`, with a
2x latent spatial upsampler. The default teacher schedule is
`[0.909375, 0.725, 0.421875, 0.0]`; the distilled student performs one denoising
transition from `0.909375` to `0.0`.

## Repository Layout

```text
configs/                         portable model and FSDP configs
examples/                        WebDataset metadata examples
packages/ltx-core/               compatible LTX core snapshot
packages/ltx-pipelines/          compatible LTX pipelines snapshot
packages/ltx-trainer/            compatible LTX trainer utilities
packages/ltx-sr-trainer/         Echo-SR datasets, losses, and trainers
scripts/train_22b.sh             LTX-2.3 22B training entry
scripts/train_19b_dmd.sh         LTX-2 19B DMD/GAN training entry
scripts/infer.sh                 one-video teacher/student inference
tools/build_train_index.py       build an index from WebDataset tar shards
```

Checkpoints, datasets, logs, and generated videos are intentionally excluded
from Git.

## Requirements

- Linux with NVIDIA GPUs
- Python 3.11 or 3.12
- CUDA 12.8-class environment
- PyTorch 2.7 or newer
- Eight H200-class GPUs are recommended for the provided FSDP config; actual
  memory depends on resolution, frame count, LoRA rank, and enabled losses
- `ffmpeg` available on `PATH`

Install with `uv`:

```bash
git clone git@github.com:xin1u/Echo-SR.git
cd Echo-SR
uv sync --all-packages --all-extras
source .venv/bin/activate
```

Or install the workspace packages into an existing CUDA environment:

```bash
python -m pip install -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer \
  -e 'packages/ltx-sr-trainer[tracking]'
```

## Model Assets

Download the matching base model, distilled LoRA, spatial upsampler, Gemma text
encoder, and LPIPS VGG weights. Put them under `checkpoints/`, or edit the four
paths in the selected YAML.

The LTX-2.3 assets are published by Lightricks at
<https://huggingface.co/Lightricks/LTX-2.3>. The example 22B config expects:

```text
checkpoints/
├── gemma-3-12b/
├── lpips_vgg.pth
├── ltx-2.3-22b-dev.safetensors
├── ltx-2.3-22b-distilled-lora-384.safetensors
└── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

For the 19B recipe, use the corresponding LTX-2 19B base, distilled LoRA, and
x2 upsampler shown in `configs/echo_sr_ltx2_19b_dmd.yaml`. Do not mix assets
between model families.

Echo-SR checkpoints are not stored in this Git repository. Large weights are
released at <https://huggingface.co/xin1u/Echo-SR>. Download the
latest published files into a separate directory:

```bash
hf download xin1u/Echo-SR --local-dir checkpoints/echo-sr
```

Use the checkpoint filename shown on the model page as
`--student-lora-path`. Keep the code commit, model revision, filename, and
checksum together in experiment records.

## Dataset

Training uses streaming WebDataset tar shards. Each sample must contain a
matching `.mp4` and `.json` member, for example:

```text
000001.mp4
000001.json
000002.mp4
000002.json
```

The JSON must provide `height`, `width`, and at least one configured caption
path. See `examples/sample_metadata.example.json`. Optional temporal fields are
`fps`, `fps_target`, `frame_num_set`, and `high_quality_frame_index`.

Create `data/train_index.json` from local shards:

```bash
python tools/build_train_index.py \
  'data/shards/*.tar' \
  --output data/train_index.json
```

The loader resizes while preserving aspect ratio and then center-crops to the
configured target dimensions. Long videos are sampled; videos shorter than the
target but not shorter than `min_frames` are mirror-padded. The public configs
use a batch size of one.

## Validate Configuration

These commands validate YAML structure without loading checkpoints or CUDA:

```bash
python packages/ltx-sr-trainer/scripts/train_stage2_sr_distill.py \
  configs/echo_sr_ltx23_22b.yaml --validate-config

python packages/ltx-sr-trainer/scripts/train_stage2_sr_distill_dmd_v2.py \
  configs/echo_sr_ltx2_19b_dmd.yaml --validate-config
```

## Training

Single-node, eight-GPU LTX-2.3 22B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29531 \
bash scripts/train_22b.sh
```

Single-node, eight-GPU LTX-2 19B DMD/GAN:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29532 \
bash scripts/train_19b_dmd.sh
```

Override a config or environment without editing a launcher:

```bash
CONFIG=configs/echo_sr_ltx23_22b.yaml \
ECHO_SR_VENV=/path/to/venv \
NPROC_PER_NODE=8 \
bash scripts/train_22b.sh
```

For multi-node training, set identical `MASTER_ADDR`, `MASTER_PORT`, `NNODES`,
and `NPROC_PER_NODE` on every node, and assign each node a distinct
`NODE_RANK=0...NNODES-1`.

To enable SwanLab, install the `tracking` extra, set
`swanlab.enable_swanlab: true` in YAML, and export the key at runtime:

```bash
export SWANLAB_API_KEY='...'
```

The key is never read from YAML. Each run writes checkpoints under
`<output_dir>/checkpoints/`, validation comparisons under
`<output_dir>/validation/`, and a copy of the resolved YAML under
`<output_dir>/run_config.yaml`.

## Inference

Run a distilled student checkpoint on one low-resolution input video:

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

Add `--teacher-lora-path` to also render the three-step teacher. The target frame
count must satisfy `8*k+1`; target height and width must be divisible by 64.
Inference is video-only and writes no audio track.

## Reproducibility Notes

- The 22B student is initialized from the official distilled LoRA and distilled
  against a frozen copy of the same LoRA using the three-step schedule.
- Only video LoRA parameters are optimized; audio LoRA parameters are frozen.
- Pixel-space supervision uses L1 and LPIPS on a sampled frame subset.
- The 19B recipe additionally trains a fake-score LoRA and a lightweight
  token-pooling discriminator.
- Config paths are interpreted relative to the repository root because the
  launchers change into that directory before invoking Python.

## Attribution and License

This repository contains a modified snapshot of LTX code. See `NOTICE.md` for
provenance and modification notes. The repository is distributed under the
included LTX-2 Community License Agreement. Model assets may carry additional
terms from their respective release pages.

Echo-SR is a research implementation associated with the broader JoyAI-Echo
ecosystem. This repository does not imply endorsement by the upstream projects.
