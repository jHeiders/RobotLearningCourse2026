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

First build takes ~10 minutes; it installs everything and puts the virtualenv on
`PATH`, so `python` inside the container is already the project's interpreter — no
`uv run`, no `activate`. On Windows, clone into the WSL2 filesystem (`~/projects/...`),
not `/mnt/c/...`; native Windows Python is not supported.

A GPU is optional; CPU-only is 2–3× slower.

If you ever change dependencies, re-run `uv sync --all-extras` (that is the only place
`uv` is still used — it installs `.venv` and keeps `uv.lock` pinned for reproducibility).

## Running

```bash
python train.py mt3                     # runs param/mt3.yaml
python train.py mt3 --seed 1
python train.py mt3 --wandb
python train.py mt3 --resume            # continue after a crash or Ctrl-C
```

Configs live in `param/`, one file per experiment: `st_reach`, `st_push`,
`st_pick_place`, `mt3`, `mt10`.

`tasks:` in a config is either a named set (`mt3`, `mt10`, defined in
`mtsac/environments.py` and taken from Meta-World itself) or an explicit list of
environment ids for a single-task run.

Each run writes `results/<id>_s<seed>/`:

| file | what |
|---|---|
| `config.yaml` | the exact settings this run used, task set resolved |
| `tb/` | TensorBoard logs (`tensorboard --logdir results`) |
| `best_model.zip` | best mean success rate seen during training |
| `final_model.zip` | the policy at the last step |
| `checkpoint/` | latest model **and** replay buffer, for `--resume` |

Checkpoints are written every `checkpoint_freq` steps (set per config, `0` disables) and
only the latest is kept, since a replay buffer runs to hundreds of megabytes. `--resume`
picks up where the run stopped; raise `total_steps` and it will extend a finished run.
It refuses to resume if the replay buffer is missing — continuing with an empty buffer
would be a different experiment, not a resume.

To score or watch a trained policy:

```bash
python play.py mt3                           # 20 episodes per task, prints success rates
python play.py st_reach --render             # same, with a viewer window per task
python play.py mt3 --model final_model --episodes 50
```

`--render` draws on the host's display, which the container is given access to, so the
window appears on your desktop like any other application. It needs a Linux graphical
session; everything else in this project runs without one.

### Expected runtimes

Measured on an NVIDIA T1000 at ~160 steps/s for single-task; MT3 and MT10 are
extrapolations, not measurements. Lower `total_steps` in the config if this does not
fit the schedule.

| Config | Steps | Wall-clock/seed |
|---|---|---|
| st_reach, st_push, st_pick_place | 2 M | ~3.5 h |
| mt3 | 6 M | ~14 h (est.) |
| mt10 | 20 M | ~3 days (est.) |

Running 2–3 jobs concurrently on one machine costs 20–30 % per-job throughput.

### Weights & Biases

One-time, on each machine:

```bash
wandb login
```

Team (`robotlearningcourse2026`) and project (`robot-learning-mtrl`) are the defaults in
`train.py`; override the team with `$WANDB_ENTITY`. W&B is opt-in per run, so runs work
without it. `WANDB_MODE=offline` records locally, then `wandb sync wandb/offline-run-*`.

Seeds of one config share a W&B group, so the dashboard stays readable. Before
launching, filter the shared project for `<id>_s<seed>` — if it is there, someone has
already run it.

## Tuning

Hyperparameters start from the values another group reported for the same benchmark
(`reference/Group19`). Everything under `sac:` in a config is passed straight to the
Stable-Baselines3 `SAC` constructor, so any SAC argument can be set there.

A new algorithm variant is a subclass of `SAC` in `mtsac/sac.py` plus an entry in
`ALGO_TABLE`; a config selects it with `algo:`. Nothing else changes.

Configs are checked when they load: an unknown key under `env:`, `sac:` or `train:`
raises immediately instead of being ignored, so a typo like `batchsize:` cannot quietly
send a three-day run off with the default value.

```bash
pytest -q          # env contract + config + end-to-end resume, ~25 s
ruff check .
```

## Meta-World v3 notes

Verified against `metaworld==3.1.1`; several of these contradict the published
documentation, and all are handled in `mtsac/`.

- Environment ids end in `-v3`. Most tutorials and papers online are v2.
- `use_one_hot` defaults to `False`. Without it a multi-task policy cannot tell the
  tasks apart — it is the single biggest lever on multi-task success.
- A seed of `0` is silently discarded (metaworld does `None if not seed else seed + idx`),
  leaving the environment unseeded. `mtsac.environments.env_seed()` maps run seeds into a
  non-zero, non-overlapping range, and gives evaluation its own window.
- Episodes end by truncation, not termination (`terminated=False` at the time limit).
  SB3 needs `TimeLimit.truncated` to keep bootstrapping the final state; the wrapper
  sets it. Omitting it corrupts SAC's value targets on nearly every episode.
- Autoreset is `SAME_STEP`, matching SB3. The terminal observation arrives in
  `final_obs` and the last step's `success` in `final_info`, which the wrapper lifts up
  so `info["success"]` is always present.
- An episode counts as a success if `info["success"]` was ever set during it, not only
  at the final step.

## Layout

```
train.py               entrypoint: config -> env -> SAC -> learn
play.py                score or watch a trained model
param/                 one YAML per experiment
mtsac/environments.py  task sets (mt3, mt10) -> vectorised Meta-World env
mtsac/wrapper.py       Gymnasium VectorEnv -> SB3 VecEnv
mtsac/sac.py           ALGO_TABLE: the algorithms a config can select
mtsac/eval.py          rollout loop, success rate per environment
mtsac/callback.py      periodic evaluation, per-task logging, best-model saving
mtsac/checkpoint.py    latest model + replay buffer, for --resume
mtsac/config.py        loads a param/*.yaml and rejects unknown keys
tests/                 env contract, config validation, end-to-end resume
```
