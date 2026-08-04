#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if [[ -n ${ECHO_SR_VENV:-} ]]; then
  source "$ECHO_SR_VENV/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT/packages/ltx-core/src:$REPO_ROOT/packages/ltx-pipelines/src:$REPO_ROOT/packages/ltx-trainer/src:$REPO_ROOT/packages/ltx-sr-trainer/src${PYTHONPATH:+:$PYTHONPATH}"
exec python packages/ltx-sr-trainer/scripts/infer_echo_sr.py "$@"
