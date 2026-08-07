#!/usr/bin/env python3
"""Echo-SR supervised fine-tuning entry point."""

import sys

import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config.yaml>")
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
