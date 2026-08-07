# ltx-sr-trainer

DMD training helpers for Echo-SR on LTX-2 19B and LTX-2.3 22B. The DMD losses
supervise the video branch; the audio LoRA stays frozen at its official
distilled-LoRA values, so the released checkpoints support joint audio-video
1-step SR.

Use the repository-level `scripts/train_dmd_19b.sh` and
`scripts/train_dmd_22b.sh` launchers rather than invoking modules directly.
