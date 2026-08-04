# Notices and Attribution

## LTX Source

This repository includes source derived from the LTX-2 project:

- Upstream repository: <https://github.com/Lightricks/LTX-2>
- Upstream model page: <https://huggingface.co/Lightricks/LTX-2.3>
- Copyright and license terms: see the root `LICENSE`

The following directories are a compatibility snapshot used by Echo-SR:

- `packages/ltx-core`
- `packages/ltx-pipelines`
- `packages/ltx-trainer`

## Echo-SR Modifications

The Echo-SR release adds or modifies:

- stage-2 SR WebDataset loading and video degradation;
- native LoRA attachment and checkpoint handling;
- three-step teacher to one-step student online distillation;
- the 19B DMD, fake-score, and token-pooling GAN training path;
- FSDP launch configuration, portable YAML files, and release-safe logging;
- teacher/student stage-2 SR inference and tiled VAE decoding.

Modified training and inference entry points carry an Echo-SR release notice in
their module docstrings. Private infrastructure paths and credentials are not
part of this distribution.

## JoyAI-Echo

JoyAI-Echo is available at <https://github.com/jd-opensource/JoyAI-Echo>.
References to JoyAI-Echo describe the surrounding research ecosystem and do not
change the license or provenance of LTX-derived code.
