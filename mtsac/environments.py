"""Build a Meta-World task set as one SB3-ready vector environment."""

from __future__ import annotations

import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the Meta-World ids with gymnasium)
from metaworld.env_dict import MT10_V3
from stable_baselines3.common.vec_env import VecMonitor

from mtsac.wrapper import GymVecToSB3

# Taken from Meta-World itself, so MT10 cannot drift out of sync with the benchmark.
# MT3 is its first three entries, which is what makes the MT3 -> MT10 one-hot a clean
# prefix and the comparison between them meaningful.
MT10_TASKS: list[str] = list(MT10_V3.keys())
MT3_TASKS: list[str] = MT10_TASKS[:3]  # reach-v3, push-v3, pick-place-v3

TASK_SETS: dict[str, list[str]] = {"mt3": MT3_TASKS, "mt10": MT10_TASKS}


def resolve_tasks(tasks: str | list[str]) -> list[str]:
    """A config's ``tasks:`` is either a named set ("mt3", "mt10") or a list of env ids.

    Single-task runs name their one task; multi-task runs name a benchmark.
    """
    if isinstance(tasks, str):
        if tasks not in TASK_SETS:
            raise ValueError(
                f"unknown task set {tasks!r}; use one of {sorted(TASK_SETS)} or a list of ids"
            )
        return list(TASK_SETS[tasks])
    return list(tasks)


def env_seed(seed: int, eval_mode: bool) -> int:
    """Map a run seed onto the seed Meta-World actually gets.

    Meta-World does ``None if not seed else seed + idx``, so seed 0 is silently dropped
    and the run comes up unseeded. Every run seed is therefore mapped into a non-zero
    window; the 1000-wide spacing keeps the per-sub-environment offsets of different
    runs, and of training vs evaluation, from overlapping.
    """
    return 1 + seed * 1000 + (500 if eval_mode else 0)


def make_env(
    tasks: list[str],
    seed: int,
    max_episode_steps: int = 500,
    use_one_hot: bool = True,
    vector_strategy: str = "async",
    envs_per_task: int = 1,
    eval_mode: bool = False,
    render_mode: str | None = None,
) -> VecMonitor:
    """One sub-environment per entry in ``tasks``, times ``envs_per_task``.

    Meta-World builds the one-hot task id from the sub-environment index, so the one-hot
    and ``tasks`` line up by construction. Repeating a task in ``tasks`` runs it in
    several environments at once (only useful with ``use_one_hot: false``).

    ``envs_per_task`` repeats every task that many times. SAC learns off-policy from one
    shared replay buffer, and a single environment feeds it one long, strongly correlated
    trajectory: consecutive transitions share an object and goal placement, so a batch
    covers almost none of the task's variation. On the harder manipulation tasks that is
    enough to make the critic diverge instead of converge. Several environments stepped in
    parallel decorrelate the buffer and cover several placements at once, at the same
    number of transitions and gradient steps. It repeats the task, not the one-hot, so the
    task id would no longer match the sub-environment index -- hence one-hot runs may not
    use it.

    Evaluation runs synchronously, one environment per task, and with its own seed window,
    so it sees different goal variations than training.
    """
    if envs_per_task < 1:
        raise ValueError(f"envs_per_task must be >= 1, got {envs_per_task}")
    if envs_per_task > 1 and use_one_hot:
        raise ValueError("envs_per_task > 1 needs use_one_hot: false; see make_env's docstring")

    # Evaluation reports one number per task and is not what training throughput depends
    # on, so it keeps the plain one-environment-per-task layout.
    copies = 1 if eval_mode else envs_per_task
    venv = gym.make_vec(
        "Meta-World/custom-mt-envs",
        envs_list=[task for task in tasks for _ in range(copies)],
        vector_strategy="sync" if eval_mode else vector_strategy,
        seed=env_seed(seed, eval_mode),
        use_one_hot=use_one_hot,
        max_episode_steps=max_episode_steps,
        reward_function_version="v2",
        # The benchmark metric is "did success happen at any point in the full episode",
        # so episodes always run to the time limit.
        terminate_on_success=False,
        render_mode=render_mode,
    )
    return VecMonitor(GymVecToSB3(venv))
