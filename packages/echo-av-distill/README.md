# echo-av-distill

1-step distillation for Echo-SR long-video super-resolution on LTX-2.3 22B. Trains a
student LoRA against the multi-step audio-video teacher produced by
[`../ltx-av-sr-trainer`](../ltx-av-sr-trainer).

The Python package is `echo_sr` (unchanged from the internal source tree so that
checkpoint metadata and config `_target_` strings keep resolving).

## Not DMD by default

`distillation.enable_dmd` controls the objective ([`src/echo_sr/training/distiller.py:148`](src/echo_sr/training/distiller.py)):

- `true` — DMD2: distribution-matching loss plus a critic update.
- `false` — teacher-trajectory distillation: the student regresses onto the teacher's
  denoised output, supervised by LPIPS + Haar wavelet + temporal-consistency losses;
  no DM loss is computed and the critic is never updated.

**Both released 1-step configs set `enable_dmd: false`,** so the published 1-step weights
are *not* a DMD student. (The short-video `ltx-sr-trainer` package in this repository is
the DMD one.)

## Contents

```
scripts/
  train.py          supervised fine-tuning entry point
  train_distill.py  1-step distillation entry point
  infer.py          single-window inference with optional TinyDecoder
src/echo_sr/
  config.py         config schema
  data/             online mp4 dataset, tar/webdataset reader, video + audio degradation
  model/            loader, rank-384 LoRA, cond_latent_2x / pixel / wave conditioning projections
  training/         EchoTrainer, EchoDistiller, strategy, perceptual losses, prompt cache
  validation/       validator + TAEHV TinyDecoder
docs/               architecture notes, resolution table, audio pipeline report,
                    LQ-proj report, pixel/wave cond-proj design, DMD distributed training
```

## Training

Use the repository-level launcher, which sets `PYTHONPATH` to this package plus the
LTX-2 **1.1** vendored snapshot (`packages/ltx-core-1.1`, `packages/ltx-trainer-1.1`):

```bash
bash scripts/train_av_distill_1k.sh          # configs/av_sr_1k_distill.yaml (audio + video)
CONFIG=configs/av_sr_1k_distill_video.yaml \
  bash scripts/train_av_distill_1k.sh        # video-only — matches the released weights
```

## Inference

```bash
bash scripts/infer_av_distill_long.sh --input /path/to/video.mp4   # long video, sliding window
python packages/echo-av-distill/scripts/infer.py \                 # single window
  --input clip.mp4 --checkpoint checkpoints/echo-sr/av-sr-1k-distill-video-step005100.safetensors
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ECHO_SR_LPIPS_DIR` | local LPIPS/VGG weight directory (avoids network download) | unset — downloads on first use |
| `SWANLAB_API_KEY` | experiment tracking, only if `swanlab.enabled: true` | unset |
