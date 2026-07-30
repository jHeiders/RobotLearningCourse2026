# Meta-World: from single-task to multi-task RL

384.195 Robot Learning course project. SAC on the Meta-World v3 benchmark, scaling from
single tasks (reach, push, pick-place) through a 3-task set to MT10.

See [PLAN.md](PLAN.md) for the schedule, compute budget and collaboration model.

## Targets

| Result | Task set | Threshold |
|---|---|---|
| R2 | reach | success > 90 % |
| R3 | push | success > 30 % |
| R4 | pick-place | success > 30 % |
| R5 | MT3 (all three) | mean success > 40 % |
| R6 | MT10 | mean success > 30 % |

## Setup

Both paths resolve the same `uv.lock`, so they produce identical environments.

**Linux, native:**

```bash
uv sync --all-extras
uv run python scripts/check_env.py
```

**Windows, or anyone wanting the exact pinned OS libraries:** open the folder in VS Code
and *Reopen in Container*. Native Windows Python is **not** supported — it uses `spawn`
rather than `fork` for multiprocessing, which changes vectorised-environment behaviour.
Use the devcontainer or WSL2.

A GPU is optional. The networks are small and the bottleneck is MuJoCo stepping on CPU,
so CPU-only is roughly 2–3× slower rather than unusable.

> If you use conda, deactivate it first — an active `VIRTUAL_ENV` makes `uv` warn and
> ignore it.

## Running

```bash
uv run python -m mtrl.train --config configs/single_task/reach.yaml --seed 0
uv run python -m mtrl.train --config configs/multi_task/mt3.yaml --seed 0 --wandb
scripts/run_sweep.sh configs/single_task/reach.yaml 0 1 2
```

Everything is config-driven. A run is fully identified by **(config file, git SHA,
seed)**, all three written to `results/<run_name>/metadata.json` and logged to W&B.

Outputs per run: `metadata.json`, `model.zip`, `final_eval.json`, TensorBoard logs
under `tb/`.

## Conventions

**Run names** are `{algo}_{taskset}_{variant}_s{seed}`, e.g. `sac_mt3_pertaskalpha_s0`.
**Before launching, filter the shared W&B project for the run name.** If it is already
there, someone has run it — that check is the entire anti-duplication mechanism.

**Learning-curve x-axis** is *total environment steps summed across all tasks*, with
steps-per-task reported alongside (`eval/steps_per_task`). Fixed before the first run,
because single-task vs multi-task sample-efficiency claims are meaningless without it.

**Success** follows the benchmark convention: an episode counts as a success if
`info["success"]` was ever set during it, not only at the final step. Per-task rates are
always logged separately — the averaged number hides exactly the inter-task interference
the project is about.

## Working together

The **core is frozen and PR-gated**: `mtrl/envs/`, `mtrl/eval/`, `mtrl/config.py`,
`mtrl/registry.py`, `mtrl/train.py`. Every number either of us produces depends on these
being stable and trusted.

**Variants are added, never edited in.** A new algorithm is a new file in `mtrl/algos/`
that registers itself, plus a config that selects it by name:

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

Two people can then add variants at the same time without touching the same file.

`results/` and `figures/` are gitignored. Configs are committed; results live in W&B.

## Meta-World v3 gotchas

Verified against `metaworld==3.1.1` — several contradict the published documentation.
They are encoded in the code and locked down by `tests/test_envs.py`.

- **Environment ids end in `-v3`.** Most tutorials and papers online are v2.
- **`use_one_hot` defaults to `False`.** The docs state MT10/MT50 append a one-hot task
  id automatically. They do not unless asked. Without it a multi-task policy cannot tell
  the tasks apart.
- **`num_tasks` is the one-hot width**, not the "parametric goal variations" the docs
  describe, and `custom-mt-envs` sets it itself — passing it raises `TypeError`. The
  goal-variation knob is `num_goals` (default 50), and it only reaches the MT1/MT10/MT50
  entry points, so we leave it at the default everywhere.
- **A seed of `0` is silently discarded** on the `custom-mt-envs` path: metaworld does
  `None if not seed else seed + idx`, so the environment comes up unseeded and the run
  is not reproducible. `mtrl.envs.make.env_seed()` maps run seeds into a non-zero,
  non-overlapping range. Since 0 is the natural default seed, this one is a trap.
- **Episodes end by truncation, not termination** (500-step limit, `terminated=False`).
  SB3 must see `TimeLimit.truncated` so it still bootstraps the value of the final
  state; the adapter sets it.
- Autoreset is already `SAME_STEP`, which matches SB3's convention — the terminal
  observation arrives in `final_obs`, and `success` for the last step is inside
  `final_info`, which the adapter lifts up so `info["success"]` is always present.

## Layout

```
configs/        one YAML per experiment — the committed run registry
mtrl/envs/      env construction + the Gymnasium->SB3 VecEnv adapter
mtrl/algos/     algorithm builders, selected by name from config
mtrl/eval/      per-task evaluation callback
mtrl/train.py   the single entrypoint
tests/          env contract tests + end-to-end smoke train (both run in CI)
scripts/        check_env.py, run_sweep.sh
```
