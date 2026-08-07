# Notices and Attribution

## LTX Source

This repository includes source derived from the LTX-2 project:

- Upstream repository: <https://github.com/Lightricks/LTX-2>
- Upstream model page: <https://huggingface.co/Lightricks/LTX-2.3>
- Copyright and license terms: see the root `LICENSE`

Echo-SR vendors **two** compatibility snapshots of that source. They are not a
version ordering — each carries code the other lacks, and they were taken at
different points for different product lines:

| Directories | Serves | Notable contents |
| --- | --- | --- |
| `packages/ltx-core`, `packages/ltx-pipelines`, `packages/ltx-trainer` | short-video, video-only DMD recipes | `fused_kernels/`, `layer_streaming.py` |
| `packages/ltx-core-1.1`, `packages/ltx-trainer-1.1` | long-video audio-video recipes | `block_streaming/`, `hdr.py`, `cond_sr_patchify.py`, `reference_audio_cond.py`, `memory_efficient_decode.py`, `sigma_tracker.py` |

Both snapshots export the same top-level Python module names (`ltx_core`,
`ltx_trainer`) and cannot be installed into a single environment. Only the 1.0
snapshot participates in the `uv` workspace; the 1.1 snapshot is reached through
`PYTHONPATH` set by the launchers in `scripts/`.

The 1.1 snapshot is unmodified apart from an invalid `target-version` value in
`packages/ltx-trainer-1.1/pyproject.toml` (`[tool.ruff]`), corrected to `py310`.

## Echo-SR Modifications

### Short-video DMD line (`packages/ltx-sr-trainer`)

- stage-2 SR WebDataset loading and video degradation;
- native LoRA attachment and checkpoint handling;
- shared WebDataset online teacher/student distillation for 19B and 22B;
- DMD distribution matching with real-score and fake-score LoRA branches;
- token-pooling GAN training and pixel-space auxiliary losses;
- FSDP launch configuration, portable YAML files, and release-safe logging;
- teacher/student stage-2 SR inference and tiled VAE decoding.

### Long-video audio-video line (`packages/ltx-av-sr-trainer`, `packages/echo-av-distill`)

- joint audio-video restoration strategy: LQ video and audio latents injected as
  channel-concatenated conditions absorbed by an expanded `patchify_proj` /
  `audio_patchify_proj`, with sampled condition-noise strength;
- audio-to-video and video-to-audio cross-attention gradient isolation below a
  configurable transformer layer;
- rank-384 LoRA coverage extended over the audio branch and the audio-video
  cross-attention gate adaLN modules;
- online audio-video dataset with GPU-side video degradation and a separate
  audio degradation pipeline;
- fixed-prompt embedding cache that removes the text encoder from the training
  and 1-step inference loops;
- sliding-window long-video inference with rank-parallel window scheduling,
  crossfade blending, and drop-first-frame i2v chaining across windows;
- 1-step distillation from a multi-step teacher with LPIPS, Haar wavelet,
  temporal-consistency, and latent-frequency losses (`enable_dmd: false`),
  alongside the optional DMD2 path;
- FSDP configuration for `BasicAVTransformerBlock` with sharded state dicts.

Modified training and inference entry points carry an Echo-SR release notice in
their module docstrings. Private infrastructure paths and credentials are not
part of this distribution.

## TinyDecoder (TAEHV)

`packages/ltx-av-sr-trainer/src/ltx_av_sr_trainer/tiny_decoder.py` and
`packages/echo-av-distill/src/echo_sr/validation/tiny_decoder.py` implement TAEHV,
a tiny autoencoder used to decode latents to pixels quickly during validation.
The architecture derives from Ollin Boer Bohan's TAESD / Seraena work
(<https://github.com/madebyollin/taehv>, <https://github.com/madebyollin/seraena>),
released under the MIT License. The released weights
(`taeltx2_3_wide.pth`) were trained by us against the LTX-2.3 video VAE.

## JoyAI-Echo

JoyAI-Echo is available at <https://github.com/jd-opensource/JoyAI-Echo>.
References to JoyAI-Echo describe the surrounding research ecosystem and do not
change the license or provenance of LTX-derived code.
