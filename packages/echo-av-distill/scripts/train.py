#!/usr/bin/env python3
"""Echo-SR supervised fine-tuning entry point.

This is the plain SFT loop (EchoTrainer) without a teacher. No released config
targets it — the published 1-step weights come from train_distill.py with
configs/av_sr_1k_distill_video.yaml. Modified for the portable JoyAI-Echo-SR
release in 2026.
"""

import sys

import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python packages/echo-av-distill/scripts/train.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from echo_sr.training.trainer import EchoTrainer

    trainer = EchoTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
