"""Build a Meta-World task set as one SB3-ready vector environment."""

from __future__ import annotations

import gymnasium as gym
import metaworld  # noqa: F401  (importing registers the Meta-World ids with gymnasium)
from metaworld.env_dict import MT10_V3
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from mtsac.wrapper import AppendTaskOneHot, GymVecToSB3

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


def env_copies(tasks: list[str], envs_per_task: int | dict[str, int]) -> list[int]:
    """How many sub-environments each entry of ``tasks`` gets.

    ``envs_per_task`` is either one number for every task or a per-task mapping; tasks the
    mapping does not name get one. Kept separate from ``make_env`` because the replay
    buffer needs the same task-to-sub-environment layout to weight its sampling by task.
    """
    if isinstance(envs_per_task, dict):
        unknown = sorted(set(envs_per_task) - set(tasks))
        if unknown:
            raise ValueError(f"envs_per_task names task(s) not in tasks: {unknown}")
        copies = [int(envs_per_task.get(task, 1)) for task in tasks]
    else:
        copies = [int(envs_per_task)] * len(tasks)
    if min(copies) < 1:
        raise ValueError(f"envs_per_task must be >= 1 for every task, got {envs_per_task}")
    return copies


def env_task_ids(tasks: list[str], envs_per_task: int | dict[str, int]) -> list[int]:
    """The task index each sub-environment runs, in ``make_env``'s ordering."""
    return [i for i, n in enumerate(env_copies(tasks, envs_per_task)) for _ in range(n)]


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
    envs_per_task: int | dict[str, int] = 1,
    eval_mode: bool = False,
    render_mode: str | None = None,
) -> VecMonitor:
    """One sub-environment per entry in ``tasks``, times ``envs_per_task``.

    ``envs_per_task`` repeats every task that many times. SAC learns off-policy from one
    shared replay buffer, and a single environment feeds it one long, strongly correlated
    trajectory: consecutive transitions share an object and goal placement, so a batch
    covers almost none of the task's variation. On the harder manipulation tasks that is
    enough to make the critic diverge instead of converge. Several environments stepped in
    parallel decorrelate the buffer and cover several placements at once, at the same
    number of transitions and gradient steps.

    It may also be a per-task mapping, which is what sets each task's share of the buffer:
    every environment contributes one transition per round, so a task's share is its
    environment count over the total. An even split gives every task the same share
    regardless of how hard it is, which on MT3 leaves pick-place untrained while reach --
    solved within 100k steps -- keeps consuming a third of the data. Tasks the mapping does
    not name get one environment.

    The one-hot task id is built here rather than by Meta-World, which derives it from the
    sub-environment index and would hand every copy of a task its own id. See
    ``AppendTaskOneHot``. Sub-environments stay grouped by task in ``tasks`` order, so a
    task's id is its index in ``tasks`` and MT3 stays a prefix of MT10.

    Evaluation runs synchronously, one environment per task, and with its own seed window,
    so it sees different goal variations than training.
    """
    # Evaluation reports one number per task and is not what training throughput depends
    # on, so it keeps the plain one-environment-per-task layout.
    copies = [1] * len(tasks) if eval_mode else env_copies(tasks, envs_per_task)
    venv = gym.make_vec(
        "Meta-World/custom-mt-envs",
        envs_list=[task for task, n in zip(tasks, copies, strict=True) for _ in range(n)],
        vector_strategy="sync" if eval_mode else vector_strategy,
        seed=env_seed(seed, eval_mode),
        use_one_hot=False,
        max_episode_steps=max_episode_steps,
        reward_function_version="v2",
        # The benchmark metric is "did success happen at any point in the full episode",
        # so episodes always run to the time limit.
        terminate_on_success=False,
        render_mode=render_mode,
    )
    env: VecEnv = GymVecToSB3(venv)
    if use_one_hot:
        task_ids = [i for i, n in enumerate(copies) for _ in range(n)]
        env = AppendTaskOneHot(env, task_ids, len(tasks))
    return VecMonitor(env)
