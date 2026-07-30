"""Environment construction: config -> SB3-ready vectorised Meta-World.

Facts this module encodes, all verified against metaworld 3.1.1 rather than the docs:

* Environment ids carry a ``-v3`` suffix.
* ``use_one_hot`` defaults to **False**. The documentation claims MT10/MT50 append a
  one-hot task id automatically; they do not unless you ask. Multi-task runs must pass
  it explicitly or the policy has no way to tell the tasks apart.
* ``num_tasks`` is the *one-hot width*, not the goal-variation count the docs describe,
  and ``custom-mt-envs`` sets it itself from ``len(envs_list)``. Passing it raises.
* The one-hot block is ordered by sub-environment index, so ``task_labels`` and the
  one-hot agree by construction.
* MT3 (reach, push, pick-place) is exactly the first three entries of MT10, so scaling
  from MT3 to MT10 keeps the same one-hot prefix.
"""

from __future__ import annotations

import gymnasium as gym
import metaworld  # noqa: F401  (registers the Meta-World ids with gymnasium)
from metaworld.env_dict import MT10_V3, MT50_V3
from stable_baselines3.common.vec_env import VecMonitor

from mtrl.config import EnvConfig
from mtrl.envs.adapter import GymVecToSB3

MT10_TASKS: list[str] = list(MT10_V3.keys())
MT50_TASKS: list[str] = list(MT50_V3.keys())
MT3_TASKS: list[str] = MT10_TASKS[:3]  # reach-v3, push-v3, pick-place-v3


def task_list(cfg: EnvConfig) -> list[str]:
    if cfg.benchmark == "MT10":
        return list(MT10_TASKS)
    if cfg.benchmark == "MT50":
        return list(MT50_TASKS)
    if cfg.benchmark != "custom":
        raise ValueError(f"unknown benchmark {cfg.benchmark!r}; use custom, MT10 or MT50")
    if not cfg.tasks:
        raise ValueError("env.tasks must be non-empty for benchmark 'custom'")
    return list(cfg.tasks)


def env_seed(run_seed: int, eval_mode: bool = False) -> int:
    """Map a run seed onto the seed actually handed to Meta-World.

    The ``custom-mt-envs`` entry point does ``None if not seed else seed + idx``, so a
    run seed of **0 is silently discarded** and the environment comes up unseeded — the
    run then cannot be reproduced, and a "three seed" experiment is really two seeds
    plus noise. Since 0 is the natural default seed, every run seed is mapped into a
    non-zero range instead.

    The 1000-wide spacing keeps the per-sub-environment offsets (``seed + idx``, up to
    50 for MT50) of different runs, and of training vs evaluation, from overlapping —
    otherwise different "seeds" would share goal sequences.
    """
    if run_seed < 0:
        raise ValueError(f"run seed must be non-negative, got {run_seed}")
    return 1 + run_seed * 1000 + (500 if eval_mode else 0)


def uses_one_hot(cfg: EnvConfig) -> bool:
    if cfg.use_one_hot is not None:
        return cfg.use_one_hot
    return len(task_list(cfg)) > 1


def make_vec_env(cfg: EnvConfig, seed: int, eval_mode: bool = False) -> VecMonitor:
    """Build the vectorised environment for training or evaluation.

    Evaluation always uses one environment per task and the synchronous strategy: the
    per-task success rate needs exactly one env per task, and eval is short enough that
    worker processes are not worth their startup cost.
    """
    tasks = task_list(cfg)
    one_hot = uses_one_hot(cfg)
    envs_per_task = 1 if eval_mode else cfg.envs_per_task

    if one_hot and envs_per_task != 1:
        # Meta-World builds the one-hot from the sub-environment index, so duplicating a
        # task across environments would hand each duplicate its own task identity.
        raise ValueError(
            "env.envs_per_task must be 1 when the one-hot task id is enabled "
            f"(got {cfg.envs_per_task} for {len(tasks)} tasks)"
        )

    kwargs = dict(
        vector_strategy="sync" if eval_mode else cfg.vector_strategy,
        seed=env_seed(seed, eval_mode),
        use_one_hot=one_hot,
        max_episode_steps=cfg.max_episode_steps,
        reward_function_version=cfg.reward_function_version,
        # Never terminate early during evaluation: the benchmark metric is "did success
        # occur at any point in the full-length episode".
        terminate_on_success=False if eval_mode else cfg.terminate_on_success,
        task_select=cfg.task_select,
        reward_normalization_method=cfg.reward_normalization_method,
        normalize_observations=cfg.normalize_observations,
    )

    if cfg.benchmark in ("MT10", "MT50"):
        venv = gym.make_vec(f"Meta-World/{cfg.benchmark}", **kwargs)
        labels = tasks
    else:
        labels = [t for t in tasks for _ in range(envs_per_task)]
        venv = gym.make_vec("Meta-World/custom-mt-envs", envs_list=labels, **kwargs)

    return VecMonitor(GymVecToSB3(venv, labels))
