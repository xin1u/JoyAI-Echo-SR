---
license: other
library_name: safetensors
tags:
  - video-super-resolution
  - ltx-video
  - dmd
  - lora
---

<p align="center">
  <img src="https://raw.githubusercontent.com/xin1u/Echo-SR/main/assets/echo-sr-hero.jpg" alt="Echo-SR video super-resolution" width="100%">
</p>

# Echo-SR

Echo-SR provides one-step DMD video super-resolution adapters for LTX-2 19B
and LTX-2.3 22B. Training and inference code is available in the
[Echo-SR GitHub repository](https://github.com/xin1u/Echo-SR).

## Released Weights

| File | Model family | Training step | Format |
| --- | --- | ---: | --- |
| `echo-sr-ltx2-19b-dmd-step18300.safetensors` | LTX-2 19B | 18,300 | BF16 LoRA adapter |
| `echo-sr-ltx2.3-22b-dmd-step04600.safetensors` | LTX-2.3 22B | 4,600 | BF16 LoRA adapter |

Download both adapters:

```bash
hf download xin1u/Echo-SR --local-dir checkpoints/echo-sr
```

Download one adapter:

```bash
hf download xin1u/Echo-SR \
  echo-sr-ltx2.3-22b-dmd-step04600.safetensors \
  --local-dir checkpoints/echo-sr
```

## Usage

Select the adapter that matches the LTX model family. Do not load the 19B
adapter into the 22B base model or the 22B adapter into the 19B base model.

```bash
git clone https://github.com/xin1u/Echo-SR.git
cd Echo-SR

bash scripts/infer.sh \
  --input-video input_lq.mp4 \
  --prompt 'A detailed cinematic scene.' \
  --output-dir outputs/inference \
  --checkpoint-path checkpoints/ltx-2.3-22b-dev.safetensors \
  --student-lora-path checkpoints/echo-sr/echo-sr-ltx2.3-22b-dmd-step04600.safetensors \
  --spatial-upsampler-path checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root checkpoints/gemma-3-12b \
  --target-height 1024 \
  --target-width 1536 \
  --num-frames 121 \
  --fps 24
```

The public release is video-only. Audio parameters are not trained and the
provided inference entry does not emit an audio track.

## Checksums

SHA256 values are recorded in `checksums.sha256`.

## License

The weights and code are subject to the LTX-2 Community License Agreement and
the additional terms of their respective base model releases. See the GitHub
repository for complete attribution and notices.
