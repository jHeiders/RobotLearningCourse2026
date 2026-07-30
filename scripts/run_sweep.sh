#!/usr/bin/env bash
# Launch one config across several seeds, sequentially, logging to results/.
#
#   scripts/run_sweep.sh configs/single_task/reach.yaml 0 1 2
#
# Check the shared W&B project for the run name before launching: if it is already
# there, someone has already run it.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <config.yaml> <seed> [seed ...]" >&2
    exit 1
fi

CONFIG="$1"
shift

export MUJOCO_GL="${MUJOCO_GL:-egl}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/results"

for SEED in "$@"; do
    LOG="$REPO_ROOT/results/$(basename "$CONFIG" .yaml)_s${SEED}.log"
    echo "=== $CONFIG seed=$SEED -> $LOG"
    uv run python -m mtrl.train --config "$CONFIG" --seed "$SEED" ${WANDB_FLAG---wandb} 2>&1 | tee "$LOG"
done
