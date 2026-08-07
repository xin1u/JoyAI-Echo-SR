# Audio-Video Super-Resolution: Training Guide

This document covers the three long-video audio-video recipes. For the
short-video, video-only DMD recipes see the main [README](../README.md).

> [!IMPORTANT]
> These recipes use the **1.1 vendored LTX snapshot**
> (`packages/ltx-core-1.1`, `packages/ltx-trainer-1.1`), which exports the same
> `ltx_core` / `ltx_trainer` module names as the 1.0 snapshot used by the DMD
> recipes. The two cannot live in one interpreter. Always launch through
> `scripts/train_av_*.sh` — they set `PYTHONPATH` so the 1.1 snapshot wins.

## The three recipes

| Config | Entry point | Launcher | Output |
| --- | --- | --- | --- |
| `av_sr_1k_multistep.yaml` | `train_online_v3.py` | `train_av_sr_1k.sh` | 1920×1152, audio + video |
| `av_sr_2k_multistep.yaml` | `train_online.py` | `train_av_sr_2k.sh` | 2560×1472, audio + video |
| `av_sr_1k_distill{,_video}.yaml` | `train_distill.py` | `train_av_distill_1k.sh` | 1920×1152, audio + video, 1 step |

All three take 1280×736 (`lq_resolution`) as input and 121 frames per sample.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_av_sr_1k.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_av_sr_2k.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_av_distill_1k.sh

# the config that produced the released 1-step weights (video-focused final phase)
CONFIG=configs/av_sr_1k_distill_video.yaml bash scripts/train_av_distill_1k.sh
```

`DRY_RUN=1` prints the resolved `PYTHONPATH` and entry point without launching.
`CONFIG`, `ACCELERATE_CONFIG`, `TRAIN_SCRIPT`, `NPROC_PER_NODE`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `ECHO_SR_VENV` are all
overridable.

## Data contract

The two dataset readers expect different layouts. Which one runs is decided by
`data.dataset_type` (`tar`) or by the presence of `data.preprocessed_data_root`.

### `OnlineAVDataset` — a flat directory of MP4s

Used by both multi-step recipes and by `av_sr_1k_distill.yaml`. Set
`data.preprocessed_data_root` to a directory holding **paired files**:

```text
data/av_train/
├── 0a3f91c2.mp4
├── 0a3f91c2_caption.json
├── 0b7e2d40.mp4
├── 0b7e2d40_caption.json
└── ...
```

The stem must match exactly and the suffix must be `_caption.json`. The caption
JSON is read with `caption_lang="en"`. Source clips should be at least
1920×1152 (the released runs used 1080p–4K sources, 5–17 s each); HQ is a centre
crop / resize to `hq_resolution`, LQ is a bicubic downscale of that HQ frame, and
degradation is applied on GPU inside the training step rather than in the
dataloader. Audio is read from the MP4 itself — a clip with no audio track is
skipped when `with_audio: true`.

Despite the field name, nothing is preprocessed: raw MP4s go in and latents are
encoded online.

### `TarVideoDataset` — WebDataset shards

Used by `av_sr_1k_distill_video.yaml` (`dataset_type: "tar"`). Each tar holds
exactly **one** `.mp4` and **one** `.json`:

```text
shard-000042.tar
├── 0a3f91c2.mp4
└── 0a3f91c2.json     {"width": 3840, "height": 2160, "fps": 25.0,
                       "caption": {"caption_en": "..."}}
```

`caption` may be a plain string or a dict; a dict is read as `caption_en`, then
`caption_zh`. Build the index the same way as the DMD recipes:

```bash
python tools/build_train_index.py 'data/shards/*.tar' \
  --output data/train_index/av_sr.json
```

and point `data.data_index_files` at the result.

### Validation clips

`validation.val_clips_dir` (distillation only) is a directory of short MP4s used
for fixed-sample previews. `validation.tiny_decoder_path` must point at the
TinyDecoder weights (`taeltx2_3_wide.pth`, published alongside the model
weights), which decode latents to pixels fast enough to run every 50 steps.

### Prompt embedding cache

The audio-video recipes train against a single fixed super-resolution prompt, so
Gemma is only needed once. On the first run the trainer encodes the prompt and
writes the cache; every later run loads it and skips the text encoder entirely,
saving roughly 14 GB of GPU memory:

```bash
export AV_SR_PROMPT_CACHE=checkpoints/prompt/sr_prompt_embeddings.pt
```

The cache stores the prompt text it was built from and is invalidated
automatically if `SR_FIXED_PROMPT` or the negative prompt changes. The
published `sr_prompt_embeddings.pt` matches the released checkpoints. The 1-step
inference script reads the same cache and never loads Gemma at all.

## How conditioning works

LQ video and audio are VAE-encoded and concatenated onto the **channel**
dimension of the noisy latent; an expanded `patchify_proj` /
`audio_patchify_proj` absorbs the extra channels. Training perturbs the
conditions with noise drawn from `[condition_noise_min, condition_noise_max]`
so the model does not lock onto one degradation strength — `0.4` for the 1K
recipe (heavy degradation, generative restoration) versus `0.1` for 2K (closer
to a faithful upscale).

Video and audio share one 48-layer transformer with cross-attention in every
layer. `cross_attn_grad_isolation_layer: 24` blocks gradients between the two
modalities in layers 0–23 while leaving 24–47 free, so early layers stay
modality-specific and only late layers learn joint representations.

Adapters are rank-384 / alpha-384 LoRA over ~40 module patterns, matching the
shape of the official LTX-2.3 distilled LoRA. The video-focused distillation
config lists 16 fewer audio-specific patterns as *trainable*, but the saved
checkpoint still carries the full audio adapter set — untrained audio modules
are merged in from the teacher at save time (see
[Distillation](#distillation-not-dmd-by-default)).

### Drop-first-frame i2v

`first_frame_conditioning_p` is the training-time half of the mechanism that
makes long-video inference work. With that probability, a sample's first-frame
tokens are replaced by clean HQ latents and excluded from the loss, so the model
learns "continue from this given frame" as well as "generate from scratch". At
inference the previous window's last frame is injected into those same slots and
then dropped before decode — see the
[README](../README.md#cross-window-continuity-drop-first-frame-i2v).

It is `0.5` in `av_sr_1k_multistep.yaml` and both distillation configs. The 2K
recipe sets it to `0.0`: that path exists for exact 2× upscaling of single
windows, not for long-video chaining.

### The 2K path

`av_sr_2k_multistep.yaml` upscales 1280×736 → 2560×1472, an exact 2×. That
requires mapping the 40×23 LQ latent grid onto the 80×46 HQ grid, which
`CondSRPatchifyProj` does with a learned spatial projection
(`ltx_core/model/transformer/cond_sr_patchify.py`). **That module exists only in
the 1.1 snapshot** — this recipe cannot run against `packages/ltx-core`.

The 2K recipe also raises `audio_loss_weight` to `10.0`, drops `batch_size` to
`1`, and sets `offload_optimizer_during_validation: true`, without which
validation at 2560×1472 will not fit.

## Distillation: not DMD by default

`distillation.enable_dmd` selects the objective, and **both released
distillation configs set it to `false`**. With the flag off,
`echo_sr/training/distiller.py` computes no distribution-matching loss and never
updates the critic. The student regresses onto the teacher's trajectory under
pixel, LPIPS, Haar wavelet, temporal-consistency, and latent-frequency losses.

Only the short-video `packages/ltx-sr-trainer` recipes are true DMD. Setting
`enable_dmd: true` re-enables the DMD2 path here, but no weights are released
for that configuration.

Loss weights differ sharply between the two branches:

| | `av_sr_1k_distill` (A+V losses) | `av_sr_1k_distill_video` (V-focused losses) |
| --- | --- | --- |
| `lpips_loss_weight` | 2.0 | 6.0 |
| `wavelet_loss_weight` | 0.0 | 8.0 |
| `temporal_loss_weight` | — | 1.0 |
| `audio_stft_loss_weight` | 1.0 | 0.0 |
| `reg_loss_weight` | 0.0 | 0.001 |
| `learning_rate` | 1e-5 | 5e-5 |
| `betas` | [0.0, 0.99] | [0.9, 0.99] |

`student_init: "from_teacher"` means the student LoRA starts as a copy of the
teacher's weights rather than from scratch.

### The released 1-step checkpoint is audio-video capable

`with_audio: false` in `av_sr_1k_distill_video.yaml` switches the **losses** to
video-focused terms and shrinks the *trainable* module list — it does not strip
the audio branch from the model or the checkpoint:

- The released run resumed from a joint audio-video distillation checkpoint
  (`enable_dmd: false`, `audio_loss_weight: 2.0`, `with_audio: true`), where the
  audio LoRA was trained. The video-focused phase then continued sharpening the
  video branch on tar-shard data.
- `_save_checkpoint` in `echo_sr/training/distiller.py` merges every
  audio/`av_ca` adapter tensor that is not in the student's `target_modules`
  from the teacher into the saved file, so the output always contains the full
  3,330-tensor set (2,136 audio-branch tensors) with the same key layout as the
  multi-step teacher.
- `infer_distill_v3_long.py` loads the audio VAE and vocoder unconditionally
  and denoises audio latents in the same single Euler step as video. There is
  no audio off-switch at inference; a silent input just yields silent output.

Treat `av-sr-1k-distill-video-step005100.safetensors` as a 1-step
**audio-video** SR model. The `-video` in the filename records the final
training phase, not a capability limit.

## Weight chain

```text
LTX-2.3 22B dev  +  distilled-lora-384-1.1 (structure only)
        │
        ├── av_sr_1k_multistep.yaml   ──►  av-sr-1k-multistep-step09900.safetensors
        │                                           │  (teacher_checkpoint)
        │                                           ▼
        └── av_sr_1k_distill_video.yaml ─►  av-sr-1k-distill-video-step005100.safetensors
```

Both are published on [Hugging Face](https://huggingface.co/xin1u/Echo-SR)
together with `tinydecoder/taeltx2_3_wide.pth` and
`prompt/sr_prompt_embeddings.pt`. Set `model.load_checkpoint` to resume from a
checkpoint; it is `null` in the published configs so a fresh run starts from the
base model.

## Distributed setup

`configs/accelerate/fsdp_av.yaml` wraps `BasicAVTransformerBlock` with
`TRANSFORMER_BASED_WRAP`, bf16 mixed precision, and `SHARDED_STATE_DICT` (the
DMD recipes use `FULL_STATE_DICT`; sharded is required at 22B with the audio
branch). It defaults to 8 processes on one node.

Multi-node: same `MASTER_ADDR` / `MASTER_PORT` / `NNODES` / `NPROC_PER_NODE`
everywhere, distinct `NODE_RANK` per node. The released runs used 48 GPUs
(6 nodes × 8).

Memory notes: gradient checkpointing is on in every recipe, the text encoder is
never resident once the prompt cache exists, and validation at 2K needs
optimizer offload. Eight H200-class GPUs is the practical floor for the 1K
recipes.

## Tracking

SwanLab is optional and disabled in all published configs. To enable it, install
the `tracking` extra, set `swanlab.enabled: true`, and pass the credential at
runtime only:

```bash
export SWANLAB_API_KEY='...'
```

Never write the key into a config file.
