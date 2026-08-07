---
license: other
library_name: safetensors
tags:
  - video-super-resolution
  - audio-video
  - ltx-video
  - dmd
  - lora
---

<p align="center">
  <img src="https://raw.githubusercontent.com/xin1u/JoyAI-Echo-SR/main/assets/echo-sr-hero.jpg" alt="Echo-SR video super-resolution" width="100%">
</p>

# JoyAI-Echo-SR

JoyAI-Echo-SR is the super-resolution sub-project of
[JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo). It provides
super-resolution adapters for the LTX model family, covering
two product lines. Training and inference code is available in the
[JoyAI-Echo-SR GitHub repository](https://github.com/xin1u/JoyAI-Echo-SR).

- **Short-video, video-only DMD** — one-step generators for LTX-2 19B and
  LTX-2.3 22B, distilled with DMD distribution matching + GAN + pixel losses.
- **Long-video audio-video SR** — 736p → 1K joint audio-video enhancement on
  LTX-2.3 22B: a multi-step teacher and a 1-step student, with sliding-window
  inference over arbitrarily long clips.

## Released Weights

| File | Line | Model family | Steps | Format |
| --- | --- | --- | ---: | --- |
| `echo-sr-ltx2-19b-dmd-step18300.safetensors` | DMD | LTX-2 19B | 18,300 | BF16 LoRA |
| `echo-sr-ltx2.3-22b-dmd-step04600.safetensors` | DMD | LTX-2.3 22B | 4,600 | BF16 LoRA |
| `av-sr-1k-multistep-step09900.safetensors` | AV SR | LTX-2.3 22B | 9,900 | BF16 LoRA |
| `av-sr-1k-distill-video-step005100.safetensors` | AV SR | LTX-2.3 22B | 5,100 | BF16 LoRA |
| `av-sr-2k-multistep-step08000.safetensors` | AV SR | LTX-2.3 22B | 8,000 | BF16 LoRA |

The two AV files form a teacher→student pair: the 1-step
`av-sr-1k-distill-video` model was distilled from the multi-step
`av-sr-1k-multistep` teacher (teacher-trajectory distillation with LPIPS, Haar
wavelet, and temporal losses — `enable_dmd: false`, so it is **not** a DMD
student). **Both checkpoints restore audio and video jointly.** The `-video`
in the student's filename refers to its final distillation phase, which used
video-focused losses; the audio branch was trained in an earlier joint
audio-video phase and is carried in full (the file is structurally identical
to the teacher — 3,330 tensors, 2,136 of them audio-branch). The 1-step
inference path denoises audio latents in the same single step as video and
muxes the enhanced track into the output.

`av-sr-2k-multistep-step08000.safetensors` is an independent multi-step model
for exact 2× upscaling (1280×736 → 2560×1472). It maps the LQ latent grid onto
the HQ grid with a learned `CondSRPatchifyProj` spatial projection and also
restores audio and video jointly.

Two auxiliary assets required by the AV configs ship alongside the weights:

| File | Purpose |
| --- | --- |
| `tinydecoder/taeltx2_3_wide.pth` | TAEHV fast latent preview decoder (validation / 1-step decode) |
| `prompt/sr_prompt_embeddings.pt` | Precomputed SR prompt embeddings — the 1-step path never loads a text encoder |

Download everything:

```bash
hf download xin1u/JoyAI-Echo-SR --local-dir checkpoints/echo-sr
```

## Usage

Select assets matching the LTX model family; never mix 19B and 22B assets.

Short-video DMD inference:

```bash
git clone https://github.com/xin1u/JoyAI-Echo-SR.git
cd JoyAI-Echo-SR

bash scripts/infer.sh \
  --input-video input_lq.mp4 \
  --prompt 'A detailed cinematic scene.' \
  --output-dir outputs/inference \
  --checkpoint-path checkpoints/ltx-2.3-22b-dev.safetensors \
  --student-lora-path checkpoints/echo-sr/echo-sr-ltx2.3-22b-dmd-step04600.safetensors \
  --spatial-upsampler-path checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root checkpoints/gemma-3-12b \
  --target-height 1024 --target-width 1536 --num-frames 121 --fps 24
```

Long-video AV super-resolution (736p input, any length):

```bash
# multi-step, audio + video output
NPROC_PER_NODE=8 bash scripts/infer_av_sr_long.sh \
  --input input_736p.mp4 \
  --checkpoint checkpoints/echo-sr/av-sr-1k-multistep-step09900.safetensors

# 1-step, audio + video output, drop-first-frame i2v chaining across windows
NPROC_PER_NODE=8 bash scripts/infer_av_distill_long.sh \
  --input input_736p.mp4 \
  --checkpoint checkpoints/echo-sr/av-sr-1k-distill-video-step005100.safetensors

# multi-step 2K (exact 2×: 1280×736 → 2560×1472)
NPROC_PER_NODE=8 bash scripts/infer_av_sr_long.sh \
  --input input_736p.mp4 \
  --checkpoint checkpoints/echo-sr/av-sr-2k-multistep-step08000.safetensors \
  --hq-width 2560 --hq-height 1472
```

See `docs/av_sr_training.md` in the GitHub repository for training recipes and
the data contract.

## Checksums

SHA256 values are recorded in `checksums.sha256`.

## License

The weights and code are subject to the LTX-2 Community License Agreement and
the additional terms of their respective base model releases. The TinyDecoder
architecture derives from MIT-licensed TAESD/TAEHV work by Ollin Boer Bohan;
the released weights were trained by us against the LTX-2.3 video VAE. See the
GitHub repository for complete attribution and notices.
