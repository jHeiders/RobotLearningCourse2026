#!/usr/bin/env bash
# Queue the remaining MT10 runs, one after another.
#
#   ./run_ablation.sh             # mt10_norm only, ~32 h -- the run the argument needs
#   ./run_ablation.sh full        # additionally the curriculum-only arm, ~64 h total
#   ./run_ablation.sh extras      # additionally the baseline to 6M and a second seed
#
# The argument is that the two unlearned tasks fail because of reward scale, not because
# they get too little data. With the baseline (results/mt10_s0) and the combined arm
# (results/mt10_norm_curr_s0) already measured, one run completes it:
#
#   mt10_norm           normalisation, no curriculum. baseline -> this isolates what the
#                       normalisation does; this -> combined isolates what the curriculum
#                       adds on top. If it alone lifts the two tasks, the diagnosis is
#                       demonstrated rather than asserted.
#
# The curriculum-only cell answers only "what does the curriculum do by itself", which the
# warm-started probe already indicated and which stops mattering if normalisation turns out
# to be sufficient. It is therefore optional, behind `full`.
#
# Start this only once the current training run has finished -- it does not check, and two
# runs on one GPU will simply halve each other's throughput. Launch it under tmux, or with
# nohup, so closing the terminal does not take the queue with it.

set -u
cd "$(dirname "$0")"

run() {
    local config=$1 seed=${2:-0}
    echo "=== $(date '+%F %T')  starting $config seed $seed"
    if python train.py "$config" --seed "$seed" --wandb; then
        echo "=== $(date '+%F %T')  finished $config seed $seed"
    else
        echo "!!! $(date '+%F %T')  FAILED $config seed $seed, continuing"
    fi
}

run mt10_norm 0

if [ "${1:-}" = "full" ] || [ "${1:-}" = "extras" ]; then
    # Only worth the 32 h if mt10_norm shows the curriculum still carries some of the work.
    run mt10_curr_scratch 0
fi

if [ "${1:-}" = "extras" ]; then
    # Neither changes the argument. The baseline has been flat for 800k steps with both
    # tasks at exactly zero, and one extra seed is reassurance rather than statistics.
    echo "=== $(date '+%F %T')  resuming mt10 seed 0 to the full 6M budget"
    python train.py mt10 --seed 0 --wandb --resume || echo "!!! mt10 resume failed"
    run mt10_norm_curr 1
fi

echo "=== $(date '+%F %T')  queue done"
