#!/usr/bin/env bash
# Echo-SR — 1-step distillation from the multi-step AV teacher.
#
# Default config is the audio-video branch. For the video-only branch that
# produced the released 1-step weights:
#   CONFIG=configs/av_sr_1k_distill_video.yaml bash scripts/train_av_distill_1k.sh
#
# Both configs set `distillation.enable_dmd: false` — teacher-trajectory
# distillation, not DMD2.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ -n ${ECHO_SR_VENV:-} ]]; then
  source "$ECHO_SR_VENV/bin/activate"
fi

CONFIG=${CONFIG:-configs/av_sr_1k_distill.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/fsdp_av.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-packages/echo-av-distill/scripts/train_distill.py}
GPUS_PER_NODE=${NPROC_PER_NODE:-8}
NUM_NODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

export PYTHONPATH="$REPO_ROOT/packages/ltx-core-1.1/src:$REPO_ROOT/packages/ltx-trainer-1.1/src:$REPO_ROOT/packages/echo-av-distill/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

printf '[Echo-SR] recipe=AV-SR-1K-distill config=%s nodes=%s gpus_per_node=%s rank=%s\n' \
  "$CONFIG" "$NUM_NODES" "$GPUS_PER_NODE" "$NODE_RANK"

if [[ -n ${DRY_RUN:-} ]]; then
  printf '[Echo-SR] PYTHONPATH=%s\n[Echo-SR] entry=%s\n' "$PYTHONPATH" "$TRAIN_SCRIPT"
  exit 0
fi

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --machine_rank "$NODE_RANK" \
  --num_machines "$NUM_NODES" \
  --num_processes "$TOTAL_PROCESSES" \
  "$TRAIN_SCRIPT" \
  "$CONFIG"
