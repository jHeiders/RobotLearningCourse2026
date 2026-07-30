# Meta-World: from single-task to multi-task RL

384.195 Robot Learning course project. SAC on the Meta-World v3 benchmark, from single
tasks (reach, push, pick-place) through a 3-task set to MT10.

| Result | Task set | Threshold |
|---|---|---|
| R2 | reach | success > 90 % |
| R3 | push | success > 30 % |
| R4 | pick-place | success > 30 % |
| R5 | MT3 (all three) | mean success > 40 % |
| R6 | MT10 | mean success > 30 % |

## Setup

1. Install [Docker](https://docs.docker.com/get-docker/) and VS Code with the
   **Dev Containers** extension.
2. Clone the repo and open the folder in VS Code.
3. *Reopen in Container* when prompted (or `F1` → "Dev Containers: Reopen in Container").

First build takes ~10 minutes. Then verify:

```bash
uv run python scripts/check_env.py    # versions, GPU, git SHA
uv run pytest -q                      # 16 passed
```

On Windows, clone into the WSL2 filesystem (`~/projects/...`), not `/mnt/c/...`. Native
Windows Python is not supported.

A GPU is optional; CPU-only is 2–3× slower.

`uv sync --all-extras` also works natively on Linux. The container and the host share
`.venv`, so switching between them makes `uv sync` recreate it.

## Running

```bash
# one run
uv run python -m mtrl.train --config configs/single_task/reach.yaml --seed 0

# one config across seeds, sequentially
scripts/run_sweep.sh configs/single_task/reach.yaml 0 1 2

# with Weights & Biases
WANDB=1 scripts/run_sweep.sh configs/single_task/reach.yaml 0 1 2
```

`mtrl.train` flags: `--config --seed --total-steps --out --wandb --wandb-project
--wandb-entity --no-eval --resume`.

Outputs land in `results/<run_name>/`: `metadata.json`, `model.zip`, `final_eval.json`,
TensorBoard logs under `tb/`.

Runs checkpoint every 200 k steps (model and replay buffer). `--resume` continues from
the latest checkpoint, and also extends a finished run if `total_steps` is raised.

### Expected runtimes

Measured on an NVIDIA T1000 at ~160 steps/s for single-task; MT3 and MT10 are
extrapolations, not measurements.

| Config | Steps | Wall-clock/seed |
|---|---|---|
| reach | 1 M | ~1.7 h |
| push, pick-place | 2 M each | ~3.5 h |
| MT3 | 3 M | ~7 h (est.) |
| MT10 | 10 M | ~35 h (est.) |

Running 2–3 jobs concurrently on one machine costs 20–30 % per-job throughput.

### Weights & Biases

One-time, on each machine:

```bash
uv run wandb login
```

Team (`robotlearningcourse2026`) and project (`robot-learning-mtrl`) are the defaults;
override with `WANDB_ENTITY`. W&B is opt-in per run, so runs work without it.
`WANDB_MODE=offline` records locally, then `wandb sync wandb/offline-run-*`.

Runs are organised as:

| field | value |
|---|---|
| name | `sac_mt3_pertaskalpha_s0` |
| group | task set (`mt3`) |
| job_type | variant |
| tags | algo, task set, variant, seed |

Config, git SHA and seed are attached to every run.

## Conventions

Run names are `{algo}_{taskset}_{variant}_s{seed}`. Before launching, filter the shared
W&B project for the run name — if it is there, someone has already run it.

Learning-curve x-axis is total environment steps summed across all tasks, with
steps-per-task alongside (`eval/steps_per_task`).

An episode counts as a success if `info["success"]` was ever set during it, not only at
the final step. Per-task rates are always logged separately alongside the average.

## Working together

Core modules are PR-gated: `mtrl/envs/`, `mtrl/eval/`, `mtrl/config.py`,
`mtrl/registry.py`, `mtrl/train.py`.

Add variants rather than editing existing ones. A new algorithm is a new file in
`mtrl/algos/` that registers itself, plus a config selecting it by name:

```python
from mtrl.registry import ALGOS

@ALGOS.register("mtsac_pertask_alpha")
def build(cfg, env, seed, tensorboard_log):
    ...
```

```yaml
algo:
  name: mtsac_pertask_alpha
```

Configs are committed; `results/` and `figures/` are gitignored and results live in W&B.

## Meta-World v3 notes

Verified against `metaworld==3.1.1`; several of these contradict the published
documentation. All are handled in code and covered by `tests/test_envs.py`.

- Environment ids end in `-v3`. Most tutorials and papers online are v2.
- `use_one_hot` defaults to `False`. MT10/MT50 do not append a one-hot task id unless
  asked, and without it a multi-task policy cannot tell the tasks apart.
- `num_tasks` is the one-hot width, not the number of goal variations, and
  `custom-mt-envs` sets it itself — passing it raises `TypeError`. The goal-variation
  knob is `num_goals` (default 50); we leave it at the default.
- A seed of `0` is silently discarded on the `custom-mt-envs` path (metaworld does
  `None if not seed else seed + idx`), leaving the environment unseeded.
  `mtrl.envs.make.env_seed()` maps run seeds into a non-zero, non-overlapping range.
- Episodes end by truncation, not termination (500-step limit, `terminated=False`). SB3
  needs `TimeLimit.truncated` to bootstrap the final state; the adapter sets it.
- Autoreset is `SAME_STEP`, matching SB3. The terminal observation arrives in
  `final_obs` and the last step's `success` in `final_info`, which the adapter lifts up
  so `info["success"]` is always present.

## Layout

```
configs/        one YAML per experiment
mtrl/envs/      env construction + Gymnasium->SB3 VecEnv adapter
mtrl/algos/     algorithm builders, selected by name from config
mtrl/eval/      per-task evaluation callback
mtrl/train.py   entrypoint
tests/          env contract tests + end-to-end smoke train (run in CI)
scripts/        check_env.py, run_sweep.sh
```
