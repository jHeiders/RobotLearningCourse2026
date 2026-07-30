# Project Plan — 384.195 Robot Learning

**From Single-Task to Multi-Task RL in Meta-World**

Two-person group. Scope of this plan: **code implementation only.** The report is out of scope
for now and gets planned separately once the experiments are producing numbers.

> **Schedule note:** the PDF states Nov 28 2025 – Jan 31 2026. Those dates are past.
> This plan assumes an ~8-week run. Confirm the real dates with the TA.

---

## 1. What the code has to produce

Presentations dropped. Implementation + the experimental results that feed the report:

| # | Result | Threshold |
|---|---|---|
| R1 | Reproducible implementation (fresh clone → identical run) | — |
| R2 | Single-task: reach | success > **90 %** |
| R3 | Single-task: push | success > **30 %** |
| R4 | Single-task: pick-place | success > **30 %** |
| R5 | Multi-task MT3 (reach, push, pick-place) | avg success > **40 %** |
| R6 | Scaling to MT10 | avg success > **30 %** |
| R7 | Comparative data: convergence, reward trends, stability, inter-task interference | — |

R7 is where marks are won — the thresholds are low, the analysis is not. Everything the code
does should be in service of producing *comparable* numbers, not just passing numbers.

---

## 2. Decisions

### 2.1 Container: **yes — devcontainer, built in week 1**

This reverses the earlier recommendation, because the second person may be on Windows.

The reasoning: on Windows, Docker Desktop runs on WSL2 anyway, so the real choice is "raw WSL2
+ uv" vs "devcontainer on WSL2" — both need WSL2 regardless. Given that, the devcontainer wins
outright, because it also buys:

- **Identical OS-level libs** (GL/EGL for MuJoCo rendering, ffmpeg) — the exact things that
  differ between a Windows box and a Linux box.
- **Linux process semantics everywhere.** Native Windows Python uses `spawn` rather than `fork`
  for multiprocessing, which changes how `SubprocVecEnv` behaves. Hard constraint for this repo:
  **Linux semantics only. Windows participates via devcontainer or WSL2, never native Python.**
  That deletes an entire class of bug we'd otherwise debug on someone else's machine.
- **It doubles as the reproducibility artifact** for R1, so this isn't extra work — it's work
  pulled forward from the end of the project to the start.

Native `uv sync` on Linux stays first-class: the Dockerfile is thin (`python:3.12-slim` +
`libgl1`/`libegl1` + `uv` + `uv sync`) and runs the identical commands. Linux person can ignore
the container day-to-day if they prefer; both paths resolve the same `uv.lock`.

**W1 checkpoint for the Windows side:** verify GPU passthrough into the container
(`--gpus all` via Docker Desktop + WSL2 + NVIDIA). If it doesn't work, **it is not a blocker** —
see §5, these nets are tiny and the bottleneck is MuJoCo on CPU, not the GPU.

### 2.2 RL framework: **Stable-Baselines3 throughout** (as chosen)

Dependency resolution verified:

```
python  3.12                     (metaworld: >=3.10,<3.14 · SB3: >=3.10)
metaworld       3.1.1            → gymnasium>=1.1, mujoco==3.3.0
stable-baselines3 2.9.0          → gymnasium<2.0,>=0.29.1, torch>=2.8   ✓ no conflict
torch           2.8+ cu12x       → sm_75 (T1000, Turing) supported      ✓
```

One gap worth knowing: SB3's `SAC` has a **single scalar entropy temperature**, where canonical
MT-SAC uses a **per-task ("disentangled") alpha**. That's a ~60-line `SAC` subclass overriding
`train()` with a `log_ent_coef` vector of shape `[n_tasks]` gathered via the one-hot. It becomes
one of the variants in §6.

### 2.3 Tracking & hosting: **public GitHub repo + shared Weights & Biases project**

W&B is the run registry — both people see each other's runs live with config, seed and git SHA
attached, which is what stops the same experiment being run twice. TensorBoard stays on as the
local fallback (SB3 writes it natively at no cost).

---

## 3. Repository structure

Follows the conventions in `pyMoPrim` / `LocalPerformanceAwareLfD` (uv, setuptools, flat
package, `scripts/`, `results/`, `tests/`).

```
RobotLearning/
├── CLAUDE.md
├── PLAN.md
├── README.md                   # setup (both paths), how to run, W&B conventions
├── pyproject.toml
├── uv.lock                     # COMMITTED → identical env everywhere
├── .python-version             # 3.12
├── .gitignore
├── .devcontainer/
│   ├── devcontainer.json       # GPU passthrough, mounts, extensions
│   └── Dockerfile              # slim: python + libgl1/libegl1 + uv sync
├── .github/workflows/ci.yml    # ruff + pytest + 2k-step smoke train
│
├── project_description/
│   └── RL_Course-Student_Project.pdf
│
├── configs/                    # the run registry. one YAML per experiment. COMMITTED.
│   ├── base.yaml
│   ├── single_task/{reach,push,pick_place}.yaml
│   └── multi_task/{mt3,mt10}.yaml
│
├── mtrl/                       # the package — CORE is frozen after W2, see §6.2
│   ├── envs/
│   │   ├── make.py             # env factories + SB3 VecEnv construction
│   │   ├── adapter.py          # ⚠ Gymnasium VectorEnv → SB3 VecEnv (highest-risk file)
│   │   └── wrappers.py         # success tracking, obs normalisation, reward shaping hooks
│   ├── algos/
│   │   ├── sac.py              # thin config → SB3 SAC
│   │   └── mtsac.py            # SAC subclass: per-task alpha
│   ├── nets/
│   │   └── extractors.py       # multi-head / task-embedding feature extractors
│   ├── eval/
│   │   └── callbacks.py        # MultiTaskEvalCallback → per-task success rate
│   ├── registry.py             # ← name → class lookup for algos / extractors / wrappers
│   ├── config.py               # dataclass config + YAML load + hash
│   └── train.py                # the single entrypoint
│
├── scripts/
│   ├── run_sweep.sh            # one config × N seeds
│   ├── check_env.py            # print versions/driver — for diagnosing machine drift
│   ├── make_figures.py         # figures from W&B export
│   └── record_video.py         # rollout videos
│
├── tests/
│   ├── test_envs.py            # obs dims, one-hot layout, seeding, autoreset semantics
│   └── test_smoke_train.py     # 2k-step train completes end-to-end
│
├── results/                    # gitignored except .gitkeep + final summary CSVs
└── figures/                    # gitignored except .gitkeep
```

One way to launch anything:

```bash
uv run python -m mtrl.train --config configs/single_task/reach.yaml --seed 0
```

A run is reproducible from **(config file, git SHA, seed)** — all three logged to W&B.

### 3.1 `registry.py` — why it exists

This file is a direct consequence of the collaboration model in §6. Because both people now
work on the same phases at the same time, they will both want to add algorithm and architecture
variants concurrently. If adding a variant means *editing* `train.py`, every variant is a merge
conflict.

So variants are **registered, not wired in**:

```python
@register_algo("mtsac_pertask_alpha")
class MTSACPerTaskAlpha(SAC): ...
```

and a config selects one by name:

```yaml
algo: mtsac_pertask_alpha
extractor: multihead
```

New variant = one new file + one new config. Core untouched. Zero conflicts.

---

## 4. Technical design and the traps

### 4.1 Meta-World v3 API (verified against current docs)

```python
env  = gym.make("Meta-World/MT1", env_name="reach-v3", seed=seed)
envs = gym.make_vec("Meta-World/MT10", vector_strategy="sync", seed=seed)
envs = gym.make_vec("Meta-World/custom-mt-envs", vector_strategy="sync",
                    envs_list=["reach-v3", "push-v3", "pick-place-v3"], seed=seed)
```

Note the **`-v3` suffix** — most tutorials and papers online are v2 and will mislead you.
Relevant kwargs: `max_episode_steps`, `terminate_on_success`, `reward_function_version`
(`"v1"`/`"v2"`), `num_tasks` (parametric goal variations, default 50), `task_select`.

Use the official `custom-mt-envs` route for MT3 — the brief links to it directly, and it gives
MT3 the same one-hot and goal-sampling semantics as MT10, which is what makes the MT3 → MT10
scaling comparison honest.

### 4.2 Trap 1 — autoreset: **checked, not a problem**

The concern was that Gymnasium 1.x defaults `VectorEnv` autoreset to NEXT_STEP while SB3 expects
SAME_STEP. Meta-World's vector entry points **already default to `AutoresetMode.SAME_STEP`**, so
the mismatch never arises.

What the adapter does still have to translate is smaller but just as silent if wrong:

* Gymnasium batches infos as `{key: array, "_key": mask}`; SB3 wants a list of per-env dicts.
* The terminal observation is in `infos["final_obs"][i]`; SB3 reads `infos[i]["terminal_observation"]`.
* **Meta-World episodes end by truncation, not termination** (500-step limit, `terminated=False`).
  SB3 needs `infos[i]["TimeLimit.truncated"]` to keep bootstrapping the final state's value.
  Miss it and SAC's targets are wrong on essentially every episode.
* `success` for the final step lives inside `final_info`, so the adapter lifts it up.

All of it is locked down by `tests/test_envs.py`.

### 4.3 ⚠ Trap 2 (the real one) — **a seed of 0 is silently discarded**

Metaworld's `custom-mt-envs` entry point does `None if not seed else seed + idx`. Seed `0` is
falsy, so it becomes `None` and the environment comes up **unseeded**. Verified: MT3 built with
seed 0 gives different observations on every construction, while seeds 1 and 7 are reproducible.

Since 0 is the natural default seed, this would have quietly voided the reproducibility
requirement and turned a "three seed" experiment into two seeds plus noise.
`mtrl.envs.make.env_seed()` maps run seeds into a non-zero range, spaced so that different runs'
per-sub-env seeds (`seed + idx`) and train-vs-eval windows never overlap.

### 4.4 ⚠ Trap 3 — the docs disagree with the package

* **`use_one_hot` defaults to `False`.** The documentation says MT10/MT50 append a one-hot task
  id automatically. They do not unless asked — a multi-task policy would have no way to tell
  the tasks apart.
* **`num_tasks` is the one-hot width**, not the "parametric goal variations" the docs describe.
  `custom-mt-envs` sets it itself from `len(envs_list)`, so passing it raises `TypeError`. The
  real knob is `num_goals` (default 50), and it only reaches the MT1/MT10/MT50 entry points —
  exposing it would silently desynchronise MT3 from MT10, so it stays at the default.
* Observation dims, verified: single task 39, MT3 42, MT10 49.
* MT3 (reach, push, pick-place) is exactly `MT10[:3]`, so the MT3 one-hot is a prefix of the
  MT10 one and the scaling comparison is clean.

### 4.5 ⚠ Trap 4 — pin the x-axis convention now

Single-vs-multi sample-efficiency claims are meaningless without it:

> Primary x-axis = **total environment steps summed across all tasks**.
> Secondary, always reported alongside = **steps per task** = total / n_tasks.

Every figure carries both. Pin it before the first run, not after.

### 4.6 Success metric

Meta-World reports success in `info["success"]` (0/1 per step). Benchmark convention is
**"success occurred at any point in the episode"** → `any()` over the episode, averaged over N
eval episodes per task. `MultiTaskEvalCallback` logs `eval/success/{task}` per task plus
`eval/success/mean`. Per-task breakdown is non-negotiable — it *is* the interference analysis
in R7.

Train with `terminate_on_success=False` (standard). Evaluating with it on would bias results.

### 4.7 SB3 wiring specifics

- `SubprocVecEnv` over the task set; `train_freq=(1, "step")` with `gradient_steps=n_envs`
  keeps update-to-data ratio at 1 (MT-SAC convention).
- Single **shared** replay buffer across tasks — that is what makes it MT-SAC.
  1 M transitions × ~49 dims float32 ≈ 0.4 GB, trivial against 62 GB.
- Net arch `[400, 400, 400]` (MT-SAC reference shape, ~1.6 M params).
- `MUJOCO_GL=egl` for headless rendering; `osmesa` as the CPU fallback if EGL misbehaves in
  the container (only affects video recording, never training).

---

## 5. Compute budget

Reference box: 20 cores, NVIDIA T1000 8 GB, 62 GB RAM. Expect 2–3 training processes per machine
concurrently.

**The GPU matters less than you'd think.** The nets are ~1.6 M params of MLP; the bottleneck is
MuJoCo stepping on CPU. A CPU-only box is maybe 2–3× slower, not 50× — which is why a Windows
GPU-passthrough failure is an annoyance, not a blocker.

**Measured**, not estimated: `configs/single_task/reach.yaml` on the T1000 (CUDA, batch 512,
4 async envs, UTD = 1) runs at **~100 steps/s** in steady state. The first log line reads higher
(301 fps) only because `learning_starts` has not elapsed and no gradient steps are running yet:

| window | steps/s |
|---|---|
| 6 k → 10 k | 105 |
| 10 k → 16 k | 98 |
| 16 k → 20 k | 100 |

That is roughly **half** the 150–250 originally assumed, so the budget below is the real one:

| Experiment | Steps | Seeds | Wall-clock/seed | Total |
|---|---|---|---|---|
| MT1 reach | 1 M | 3 | ~2.8 h | ~8 h |
| MT1 push | 2 M | 3 | ~5.6 h | ~17 h |
| MT1 pick-place | 2 M | 3 | ~5.6 h | ~17 h |
| MT3 | 3 M | 3 | ~8.3 h | ~25 h |
| MT3 variants | 3 M | 2 arms × 3 | ~8.3 h | ~50 h |
| **MT10** | **10 M** | **2–3** | **~40 h** † | **80–120 h** |

† MT10 uses batch 1280 and 10 envs, so expect ~70 steps/s rather than 100. Measure it on the
first MT10 launch rather than trusting this extrapolation.

Total ≈ **200–240 h** of sequential compute. Across two machines running 2 concurrent jobs each
that is a few days of wall-clock — comfortable inside 8 weeks, but only if MT10 starts on time.

**MT10 is ~1.7 days per seed and cannot be parallelised within a seed.** Three seeds across two
machines is two sequential slots, ~3.5 days. This is the binding constraint of the project.

Levers if it gets tight, in the order I'd pull them: 2 MT10 seeds instead of 3; UTD below 1 for
MT10 only (`algo.gradient_steps` < n_envs); 1 M steps/task rather than 2 M; and accepting the
30 % threshold rather than chasing published numbers. Note that running 2–3 jobs concurrently on
one box costs perhaps 20–30 % per-job throughput — it is still a net win.

---

## 6. Collaboration model

### 6.1 Both people do every phase; they differ by *variant*, not by *task*

Multi-task is harder than single-task and depends on it, so it cannot be a parallel track — it
has to come after. Both people therefore move through the same phases together, and the
parallelism is on the **variant axis**: different reward tuning, different RL approach,
different conditioning architecture.

| Phase | Both do | A explores | B explores |
|---|---|---|---|
| **P0 — infra** | joint, pair on the core | — | — |
| **P1 — single-task** | reach, push, pick-place | SAC hyperparameter recipe (lr, batch, UTD, net width) | reward tuning: `reward_function_version`, shaping wrapper, `terminate_on_success`, obs normalisation |
| **P2 — MT3** | multi-task on the 3 tasks | shared backbone + one-hot, per-task α | multi-head / task-embedding extractor |
| **P3 — MT10** | scale the winner | winning recipe × seeds | runner-up as the ablation arm |

Each phase ends in a **bake-off**: compare the two variants on the same metric, pick one to
carry forward, record the decision in a GitHub issue. The losing arm is *not* wasted — it is
exactly the comparative data R7 needs. Optionally A takes SAC and B takes PPO in P1 as the
algorithm axis; treat PPO as an analysis arm rather than a threshold-clearing one, since PPO on
Meta-World is much weaker than SAC.

### 6.2 How duplication is avoided when both do everything

Ownership-by-directory no longer works, so the seam moves:

1. **The core is frozen after P0 and PR-gated.** `mtrl/envs/`, `mtrl/eval/`, `mtrl/config.py`,
   `mtrl/train.py`, `mtrl/registry.py` change only via PR reviewed by the other person. Every
   number either person produces depends on these files being trusted and stable.
2. **Variants are added, never edited in.** A variant is a new file in `mtrl/algos/` or
   `mtrl/nets/` plus a new config YAML, registered by name (§3.1). Two people adding variants
   simultaneously touch disjoint files.
3. **W&B is the run registry.** Naming convention, documented in the README:
   `{algo}_{taskset}_{variant}_s{seed}` → `mtsac_mt3_pertaskalpha_s0`. Before launching, filter
   the project for that name. If it exists, don't run it.
4. **Configs committed, results not.** `results/` and `figures/` gitignored except `.gitkeep`
   (same pattern as `LocalPerformanceAwareLfD`). Anyone can reproduce anyone's run from the
   committed config.
5. **GitHub Issues, one per experiment or variant, assigned.** Labels: `infra`, `single-task`,
   `multi-task`, `analysis`. This is the "who is running what right now" board.
6. **Feature branches + PR review by the other person**, `main` always green (CI: ruff + pytest
   + 2k-step smoke train). On a two-person project both people need to be able to explain the
   other's results, and PR review is the cheapest way to get there.

### 6.3 Environment parity

`uv sync` from the committed `uv.lock` + `.python-version`, inside the devcontainer if on
Windows. `scripts/check_env.py` prints Python/torch/CUDA/metaworld/SB3 versions and driver —
run it first whenever two machines' results disagree. Success rates should match across
machines; wall-clock numbers will not if the hardware differs, so never compare throughput
across people.

---

## 7. Implementation phases

| Phase | Work | Done when |
|---|---|---|
| **P0 — infra** ✅ *done* | Repo, uv env, devcontainer, env layer + adapter, registry, eval callback, 14 tests, CI. Throughput measured (§5). | ✅ `pytest` green; smoke train produces artefacts; ~100 steps/s measured. Remaining: `git init` + GitHub repo + W&B project, and a devcontainer build on the second machine |
| **P1 — single-task** (~2 wks) | reach, push, pick-place × 3 seeds. A and B on their variant axes. Bake-off → frozen baseline recipe. | **reach > 90 %**, push & pick-place > 30 %. If reach won't clear 90 %, the infra is wrong — stop and fix, don't tune |
| **P2 — MT3** (~2 wks) | Multi-task on the 3 tasks. A: per-task α. B: multi-head / task embedding. Bake-off. | **MT3 > 40 %** |
| **P3 — MT10** (~2 wks) | **Launch early — 1–2 days per seed.** Winner scaled to MT10, runner-up as ablation arm. | **MT10 > 30 %**, per-task breakdown collected |
| **P4 — consolidation** (~0.5 wk) | Figures from W&B, fresh-clone + container reproducibility check, summary CSVs into `results/`. | clean clone → `uv sync` → reproduces a curve |

Report planning happens after P2, once there are real numbers to write about.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ~~Autoreset semantics~~ | — | ✅ Not an issue: Meta-World already defaults to SAME_STEP (§4.2) |
| ~~Seed 0 silently unseeded~~ | ~~Reproducibility void~~ | ✅ Found and fixed in P0 (§4.3), regression-tested |
| Throughput ~100 steps/s, half the estimate | MT10 is ~1.7 days/seed | ✅ Measured (§5); budget rebuilt around it; pull the levers listed there early, not late |
| Windows GPU passthrough fails | Slower runs on one machine | Not a blocker — CPU-bound workload (§5); fall back to CPU torch |
| pick-place doesn't reach 30 % | Missed R4 | Hardest single task; budget 2 M steps and extra tuning in P1 |
| MT10 started too late | Missed R6 outright | Start of P3 is a hard deadline; 1–2 days per seed |
| SB3 internals shift under the `SAC` subclass | Per-task α breaks | Pin SB3 exactly in `uv.lock`; don't upgrade mid-project |
| Both add the same variant | Wasted days | Registry + Issues + W&B run registry (§6.2) |

---

## 9. Immediate next steps

P0 is built and verified. What is left:

1. `git init`, create the public GitHub repo, add the second member. *(Not done — repo-state
   changes are left to you.)*
2. Create the shared W&B project (`robot-learning-mtrl`) and set `WANDB_API_KEY` on both boxes.
3. Second member: build the devcontainer and run `uv run pytest -q` + `scripts/check_env.py`.
   Confirm GPU passthrough, or confirm CPU-only and accept ~2–3× slower.
4. Launch the first real run — `configs/single_task/reach.yaml`, 3 seeds. Clearing **90 %** is
   the signal the whole stack is sound; until it does, tune nothing else.
5. Measure MT10 throughput on its first launch and correct the † row in §5.
