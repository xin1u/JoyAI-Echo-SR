#!/usr/bin/env bash
# Echo-SR — 1-step long-video inference.
#
# Same window schedule as the multi-step path, plus drop-first-frame i2v
# chaining: each window after the first is conditioned on the previous window's
# last frame, which is then dropped before decode so no frame is emitted twice.
# A single denoising step, no CFG. Prompt embeddings come from a precomputed
# cache, so no text encoder is loaded; override its location with
# AV_SR_PROMPT_CACHE or --prompt-cache.
#
# Usage:
#   bash scripts/infer_av_distill_long.sh --input clip.mp4 [--output-dir out/]
#   NPROC_PER_NODE=4 bash scripts/infer_av_distill_long.sh --input clip.mp4
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ -n ${ECHO_SR_VENV:-} ]]; then
  source "$ECHO_SR_VENV/bin/activate"
fi

GPUS_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29511}

export PYTHONPATH="$REPO_ROOT/packages/ltx-core-1.1/src:$REPO_ROOT/packages/ltx-trainer-1.1/src:$REPO_ROOT/packages/ltx-av-sr-trainer/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

exec torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --master_port "$MASTER_PORT" \
  packages/ltx-av-sr-trainer/scripts/infer_distill_v3_long.py "$@"
