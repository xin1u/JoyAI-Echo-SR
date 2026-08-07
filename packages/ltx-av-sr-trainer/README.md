# ltx-av-sr-trainer

Long-video **audio-video** super-resolution / restoration training and inference for
LTX-2.3 22B. This is the multi-step teacher branch of Echo-SR; the 1-step distilled
student lives in [`../echo-av-distill`](../echo-av-distill).

Unlike the short-video `ltx-sr-trainer` package (DMD, LTX-2 1.0 snapshot),
everything here targets the **LTX-2 1.1 snapshot** vendored as
`packages/ltx-core-1.1` and `packages/ltx-trainer-1.1`. The two snapshots share Python
top-level module names (`ltx_core`, `ltx_trainer`) and therefore **cannot be imported in
the same process** — always launch through the repository-level shell scripts, which set
`PYTHONPATH` correctly.

## Contents

```
scripts/
  train.py                  shared SRTrainer + monkey-patch helpers (imported by both online scripts)
  train_online_v3.py        736→1K AV restoration, on-the-fly degradation
  train_online.py           736→2K AV restoration (CondSRPatchifyProj position mapping)
  infer_sr_long.py          multi-step sliding-window long-video inference (audio + video)
  infer_distill_v3_long.py  1-step sliding-window long-video inference (audio + video)
src/ltx_av_sr_trainer/
  strategy.py               AV restoration training strategy (LQ latent conditioning)
  online_av_dataset.py      raw mp4 + caption json, degradation applied online
  raw_av_dataset.py         webdataset/tar reader
  dataset.py                preprocessed-latent reader
  native_lora.py            rank-384 LoRA injection over ~40 module patterns
  mergeable_lora.py         merge LoRA into base weights for inference
  perceptual_losses.py      LPIPS, Haar wavelet, multi-resolution STFT
  tiny_decoder.py           TAEHV fast latent preview decoder
  degradation/              video + audio degradation pipelines
docs/av_architecture.md     LTX-2.3 architecture walkthrough (transformer / VAE / conditioning)
```

## Training

Use the repository-level launchers rather than calling these scripts directly:

```bash
bash scripts/train_av_sr_1k.sh    # 736→1K, configs/av_sr_1k_multistep.yaml
bash scripts/train_av_sr_2k.sh    # 736→2K, configs/av_sr_2k_multistep.yaml
```

Both default to 8 GPUs (`NPROC_PER_NODE`) with FSDP wrapping `BasicAVTransformerBlock`.

## Inference

```bash
bash scripts/infer_av_sr_long.sh --input /path/to/video.mp4
```

Long videos are cut on shot boundaries. Each 241-frame shot is covered by two
121-frame windows (`[start, start+121]` and `[start+120, start+241]`), so windows
overlap by one frame inside a shot and not at all across shots. Windows are
distributed across ranks, gathered on rank 0, and crossfaded where they meet.

`infer_sr_long.py` (multi-step) denoises each window independently.
`infer_distill_v3_long.py` (1-step) additionally chains windows with
**drop-first-frame i2v**: the previous window's last-frame latent is written into
the first `H×W` token slots, `denoise_mask` is zeroed there so those tokens are
treated as given, the model's prediction for them is re-pinned after every step,
and then `clear_conditioning()` drops them before decode — the conditioning frame
is a duplicate of one the previous window already emitted, so each window
contributes 120 new frames. `first_frame_conditioning_p` in the training configs
is the training-time counterpart of the same formulation.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `AV_SR_PROMPT_CACHE` | precomputed prompt-embedding cache for the 1-step path | `checkpoints/prompt/sr_prompt_embeddings.pt` |
| `ECHO_SR_LPIPS_DIR` | local LPIPS/VGG weight directory (avoids network download) | unset — downloads on first use |
| `SWANLAB_API_KEY` | experiment tracking, only if `swanlab.enabled: true` | unset |
| `MAX_WINDOWS` | cap sliding windows during inference (debugging) | `0` (all) |
