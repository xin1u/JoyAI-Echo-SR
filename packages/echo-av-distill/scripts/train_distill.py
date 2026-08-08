#!/usr/bin/env python3
"""Echo-SR 1-step distillation entry point.

Whether this run is a true DMD2 distillation or a teacher-trajectory distillation
is decided by `distillation.enable_dmd` in the config. The released 1-step weights
were trained with `enable_dmd: false` (teacher trajectory + perceptual / wavelet /
temporal losses, no critic update).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_av_distill_1k.sh

    # the config that produced the released 1-step weights
    CONFIG=configs/av_sr_1k_distill_video.yaml bash scripts/train_av_distill_1k.sh

Modified for the portable JoyAI-Echo-SR release in 2026.
"""

import sys

import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_distill.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from echo_sr.training.distiller import EchoDistiller

    distiller = EchoDistiller(config)
    distiller.setup()
    distiller.train()


if __name__ == "__main__":
    main()
