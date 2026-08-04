#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ -n ${ECHO_SR_VENV:-} ]]; then
  source "$ECHO_SR_VENV/bin/activate"
fi

CONFIG=${CONFIG:-configs/echo_sr_ltx23_22b_dmd.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/fsdp_8gpu.yaml}
GPUS_PER_NODE=${NPROC_PER_NODE:-8}
NUM_NODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

export PYTHONPATH="$REPO_ROOT/packages/ltx-core/src:$REPO_ROOT/packages/ltx-pipelines/src:$REPO_ROOT/packages/ltx-trainer/src:$REPO_ROOT/packages/ltx-sr-trainer/src${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

printf '[Echo-SR] recipe=LTX-2.3-22B-DMD config=%s nodes=%s gpus_per_node=%s rank=%s\n' \
  "$CONFIG" "$NUM_NODES" "$GPUS_PER_NODE" "$NODE_RANK"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --machine_rank "$NODE_RANK" \
  --num_machines "$NUM_NODES" \
  --num_processes "$TOTAL_PROCESSES" \
  packages/ltx-sr-trainer/scripts/train_stage2_sr_dmd_webdataset.py \
  "$CONFIG"
