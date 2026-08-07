#!/usr/bin/env bash
# Echo-SR — multi-step long-video audio-video inference.
#
# Sliding window over the whole clip: two 121-frame windows per 241-frame shot,
# distributed across ranks and crossfaded on rank 0. Each window is denoised
# independently — the drop-first-frame i2v chaining lives in the 1-step path
# (scripts/infer_av_distill_long.sh).
#
# Usage:
#   bash scripts/infer_av_sr_long.sh --input clip.mp4 [--output-dir out/]
#   NPROC_PER_NODE=4 bash scripts/infer_av_sr_long.sh --input clip.mp4
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ -n ${ECHO_SR_VENV:-} ]]; then
  source "$ECHO_SR_VENV/bin/activate"
fi

GPUS_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29510}

export PYTHONPATH="$REPO_ROOT/packages/ltx-core-1.1/src:$REPO_ROOT/packages/ltx-trainer-1.1/src:$REPO_ROOT/packages/ltx-av-sr-trainer/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

exec torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --master_port "$MASTER_PORT" \
  packages/ltx-av-sr-trainer/scripts/infer_sr_long.py "$@"
