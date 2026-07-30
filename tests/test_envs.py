"""Environment-layer contract tests.

These exist because the failure modes they cover are silent: a wrong one-hot width or a
missing ``TimeLimit.truncated`` flag degrades learning without raising anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtrl.config import EnvConfig
from mtrl.envs.adapter import split_infos
from mtrl.envs.make import MT3_TASKS, MT10_TASKS, env_seed, make_vec_env, task_list

SHORT_EPISODE = 5


def _cfg(**kwargs) -> EnvConfig:
    base = dict(vector_strategy="sync", max_episode_steps=SHORT_EPISODE)
    base.update(kwargs)
    return EnvConfig(**base)


def test_mt3_is_the_mt10_prefix():
    """The MT3 -> MT10 scaling story depends on this."""
    assert MT3_TASKS == MT10_TASKS[:3]
    assert MT3_TASKS == ["reach-v3", "push-v3", "pick-place-v3"]
    assert len(MT10_TASKS) == 10


def test_single_task_has_no_one_hot():
    env = make_vec_env(_cfg(tasks=["reach-v3"]), seed=0)
    try:
        assert env.num_envs == 1
        assert env.observation_space.shape == (39,)
        assert env.observation_space.dtype == np.float32
    finally:
        env.close()


def test_envs_per_task_scales_single_task_only():
    env = make_vec_env(_cfg(tasks=["reach-v3"], envs_per_task=3), seed=0)
    try:
        assert env.num_envs == 3
        assert env.observation_space.shape == (39,)
    finally:
        env.close()

    with pytest.raises(ValueError, match="envs_per_task"):
        make_vec_env(_cfg(tasks=MT3_TASKS, envs_per_task=2), seed=0)


def test_multi_task_one_hot_width_and_ordering():
    cfg = _cfg(tasks=MT3_TASKS)
    env = make_vec_env(cfg, seed=0)
    try:
        assert env.num_envs == 3
        assert env.observation_space.shape == (39 + 3,)
        obs = env.reset()
        one_hot = obs[:, -3:]
        # The one-hot must be the identity: sub-env i runs task i, which is what the
        # per-task success logging assumes.
        assert np.array_equal(one_hot, np.eye(3, dtype=np.float32))
        assert env.get_attr("task_labels")[0] == MT3_TASKS
    finally:
        env.close()


def test_terminal_transition_is_sb3_shaped():
    """Meta-World episodes end by truncation; SB3 must be told so it keeps bootstrapping."""
    cfg = _cfg(tasks=MT3_TASKS)
    env = make_vec_env(cfg, seed=0)
    try:
        env.reset()
        for _ in range(SHORT_EPISODE):
            obs, rewards, dones, infos = env.step(
                np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32)
            )
        assert dones.all(), "episode should end at max_episode_steps"
        for info in infos:
            assert info["TimeLimit.truncated"] is True
            assert "terminal_observation" in info
            assert info["terminal_observation"].shape == env.observation_space.shape
            assert info["terminal_observation"].dtype == np.float32
            # Lifted out of Meta-World's final_info so success is present on every step.
            assert "success" in info
            assert "episode" in info  # VecMonitor
        assert obs.shape == (env.num_envs,) + env.observation_space.shape
    finally:
        env.close()


def test_success_key_present_on_normal_steps():
    env = make_vec_env(_cfg(tasks=MT3_TASKS), seed=0)
    try:
        env.reset()
        _, _, dones, infos = env.step(
            np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32)
        )
        assert not dones.any()
        for i, info in enumerate(infos):
            assert "success" in info
            assert info["task"] == MT3_TASKS[i]
            assert info["task_idx"] == i
    finally:
        env.close()


def _first_obs(seed: int, **cfg_kwargs) -> np.ndarray:
    env = make_vec_env(_cfg(**cfg_kwargs), seed=seed)
    try:
        return env.reset().copy()
    finally:
        env.close()


def test_seeding_is_reproducible():
    """Seed 0 is the regression case: Meta-World treats a falsy seed as 'unseeded'."""
    for seed in (0, 1):
        assert np.allclose(
            _first_obs(seed, tasks=MT3_TASKS), _first_obs(seed, tasks=MT3_TASKS)
        ), f"seed {seed} is not reproducible"
    assert not np.allclose(_first_obs(0, tasks=MT3_TASKS), _first_obs(1, tasks=MT3_TASKS))


def test_seeds_do_not_overlap_between_runs_or_eval():
    """Distinct run seeds must not share per-sub-env seeds, or they are not independent."""
    max_sub_envs = 50
    windows = [
        range(base, base + max_sub_envs)
        for seed in range(4)
        for base in (env_seed(seed, False), env_seed(seed, True))
    ]
    seen: set[int] = set()
    for window in windows:
        assert not seen & set(window)
        seen |= set(window)
    assert env_seed(0, False) != 0  # the whole point

    with pytest.raises(ValueError, match="non-negative"):
        env_seed(-1)


def test_task_list_resolves_benchmarks():
    assert task_list(EnvConfig(benchmark="MT10")) == MT10_TASKS
    assert len(task_list(EnvConfig(benchmark="MT50"))) == 50
    with pytest.raises(ValueError, match="unknown benchmark"):
        task_list(EnvConfig(benchmark="MT7"))


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
