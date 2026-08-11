"""Environment contract tests.

These failures are all silent: a wrong one-hot width, a missing ``TimeLimit.truncated``
flag or a dropped seed degrades learning without raising anything. You would only notice
days into a run, as a number that is worse than it should be.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtsac.environments import (
    MT3_TASKS,
    MT10_TASKS,
    env_seed,
    make_env,
    resolve_tasks,
)
from mtsac.wrapper import split_infos

SHORT_EPISODE = 5


def _env(tasks, seed=0, **kwargs):
    return make_env(
        tasks, seed=seed, max_episode_steps=SHORT_EPISODE, vector_strategy="sync", **kwargs
    )


def test_mt3_is_the_mt10_prefix():
    """The MT3 -> MT10 scaling story depends on this."""
    assert MT3_TASKS == MT10_TASKS[:3]
    assert MT3_TASKS == ["reach-v3", "push-v3", "pick-place-v3"]
    assert len(MT10_TASKS) == 10


def test_resolve_tasks_accepts_named_sets_and_lists():
    assert resolve_tasks("mt3") == MT3_TASKS
    assert resolve_tasks("mt10") == MT10_TASKS
    assert resolve_tasks(["reach-v3"]) == ["reach-v3"]
    with pytest.raises(ValueError, match="unknown task set"):
        resolve_tasks("mt7")


def test_single_task_has_no_one_hot():
    env = _env(["reach-v3"], use_one_hot=False)
    try:
        assert env.num_envs == 1
        assert env.observation_space.shape == (39,)
        assert env.observation_space.dtype == np.float32
    finally:
        env.close()


def test_multi_task_one_hot_width_and_ordering():
    env = _env(MT3_TASKS)
    try:
        assert env.num_envs == 3
        assert env.observation_space.shape == (39 + 3,)
        obs = env.reset()
        # Sub-environment i must run task i: per-task logging assumes it.
        assert np.array_equal(obs[:, -3:], np.eye(3, dtype=np.float32))
    finally:
        env.close()


def test_one_hot_is_shared_by_every_copy_of_a_task():
    """``envs_per_task`` repeats the task, not the task id, so MT3 stays three wide."""
    env = _env(MT3_TASKS, envs_per_task=2)
    try:
        assert env.num_envs == 6
        assert env.observation_space.shape == (39 + 3,)
        obs = env.reset()
        # Copies stay grouped by task, in `tasks` order.
        assert np.array_equal(obs[:, -3:], np.repeat(np.eye(3, dtype=np.float32), 2, axis=0))
    finally:
        env.close()


def test_envs_per_task_accepts_a_per_task_split():
    """The split is what sets each task's share of the replay buffer."""
    env = _env(MT3_TASKS, envs_per_task={"reach-v3": 1, "push-v3": 2, "pick-place-v3": 3})
    try:
        assert env.num_envs == 6
        obs = env.reset()
        expected = np.repeat(np.eye(3, dtype=np.float32), [1, 2, 3], axis=0)
        assert np.array_equal(obs[:, -3:], expected)
    finally:
        env.close()

    with pytest.raises(ValueError, match="not in tasks"):
        _env(MT3_TASKS, envs_per_task={"door-open-v3": 2})


def test_one_hot_survives_the_terminal_observation():
    """SAC bootstraps from the terminal observation, so it needs the id too."""
    env = _env(MT3_TASKS, envs_per_task=2)
    try:
        env.reset()
        for _ in range(SHORT_EPISODE):
            _, _, dones, infos = env.step(
                np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
            )
        assert dones.all()
        terminal = np.stack([info["terminal_observation"] for info in infos])
        assert terminal.shape == (6, 39 + 3)
        assert np.array_equal(terminal[:, -3:], np.repeat(np.eye(3, dtype=np.float32), 2, axis=0))
    finally:
        env.close()


def test_terminal_transition_is_sb3_shaped():
    """Meta-World episodes end by truncation; SB3 must be told so it keeps bootstrapping."""
    env = _env(MT3_TASKS)
    try:
        env.reset()
        for _ in range(SHORT_EPISODE):
            obs, _, dones, infos = env.step(
                np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
            )
        assert dones.all(), "episode should end at max_episode_steps"
        for info in infos:
            assert info["TimeLimit.truncated"] is True
            assert info["terminal_observation"].shape == env.observation_space.shape
            assert info["terminal_observation"].dtype == np.float32
            assert "success" in info  # lifted out of final_info
            assert "episode" in info  # VecMonitor
        assert obs.shape == (env.num_envs, *env.observation_space.shape)
    finally:
        env.close()


def test_success_key_present_on_normal_steps():
    env = _env(MT3_TASKS)
    try:
        env.reset()
        _, _, dones, infos = env.step(
            np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
        )
        assert not dones.any()
        assert all("success" in info for info in infos)
    finally:
        env.close()


def _first_obs(seed: int) -> np.ndarray:
    env = _env(MT3_TASKS, seed=seed)
    try:
        return env.reset().copy()
    finally:
        env.close()


def test_seeding_is_reproducible():
    """Seed 0 is the regression case: Meta-World treats a falsy seed as 'unseeded'."""
    for seed in (0, 1):
        assert np.allclose(_first_obs(seed), _first_obs(seed)), f"seed {seed} not reproducible"
    assert not np.allclose(_first_obs(0), _first_obs(1))


def test_seeds_do_not_overlap_between_runs_or_eval():
    """Distinct run seeds must not share per-sub-env seeds, or they are not independent."""
    max_sub_envs = 50
    seen: set[int] = set()
    for seed in range(4):
        for eval_mode in (False, True):
            window = set(range(env_seed(seed, eval_mode), env_seed(seed, eval_mode) + max_sub_envs))
            assert not seen & window
            seen |= window
    assert env_seed(0, False) != 0  # the whole point


def test_split_infos_handles_masks_and_nesting():
    infos = {
        "success": np.array([1.0, 0.0]),
        "_success": np.array([True, False]),
        "final_info": {"success": np.array([1.0, 0.0]), "_success": np.array([True, False])},
        "_final_info": np.array([True, False]),
    }
    out = split_infos(infos, 2)
    assert out[0]["success"] == 1.0
    assert "success" not in out[1]  # masked off
    assert out[0]["final_info"]["success"] == 1.0
    assert "final_info" not in out[1]
