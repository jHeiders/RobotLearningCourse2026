#!/usr/bin/env bash
# Queue every MT10 run still outstanding, one after another. ~98 h total on one T1000.
#
#   tmux new -s ablation                # or nohup; closing the terminal must not kill it
#   ./run_ablation.sh
#
# The proposed method is mt10_norm_curr, which ran to 9M. Every other arm is a comparison
# against it and therefore has to be measured at the same 9M budget -- an arm stopped at 6M
# cannot answer "would it have broken through later", and the combined arm's own
# peg-insert-side was still at exactly 0.00 at 2.7M. All four configs are set to
# total_steps 9000000 with patience null; see the comment in each.
#
# What is still missing, in the order this runs them:
#
#   mt10_curr           The empty cell of the 2x2: curriculum, no normalisation, from
#                       scratch. results/mt10_curr_warmstart_probe_s0 tested the same mechanism but was
#                       warm-started from the baseline's 2.2M checkpoint, so it opens with
#                       eight tasks already at 1.00 and is not comparable. Without this cell
#                       the ablation cannot answer whether the normalisation contributes at
#                       all, given that mt10_norm alone did nothing.
#
#   mt10_norm           Measured to 5.9M, both hard tasks at exactly 0.00 throughout.
#                       Resumed here to 9M so that "0.00" is a statement about the same
#                       budget the proposed method got.
#
#   mt10                The baseline, stopped by hand at 2.3M after 23 flat evaluations.
#                       Same reason, and it is the control the other three are read against.
#
# --resume is passed to all three. train.py ignores it when the run has no checkpoint, so
# mt10_curr starts from zero on the first pass, and re-running this script after a
# crash or a reboot picks every arm up where it stopped instead of restarting it.
#
# Afterwards: python report/make_figures.py, with BUDGET set to 9.0.

set -u
cd "$(dirname "$0")"

run() {
    local config=$1 seed=${2:-0}
    echo "=== $(date '+%F %T')  starting $config seed $seed"
    if python train.py "$config" --seed "$seed" --wandb --resume; then
        echo "=== $(date '+%F %T')  finished $config seed $seed"
    else
        echo "!!! $(date '+%F %T')  FAILED $config seed $seed, continuing"
    fi
}

run mt10_curr 0           # ~47 h from zero
run mt10_norm 0           # ~16 h, 5.9M -> 9M
run mt10 0                # ~35 h, 2.3M -> 9M

echo "=== $(date '+%F %T')  queue done"
