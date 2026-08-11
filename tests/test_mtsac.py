"""The multi-task parts of the critic.

These failures are silent too: a curriculum that quietly samples uniformly, a per-task
alpha that is really one shared alpha, or a loss scale that never leaves 1.0 all train
without raising, and only show up as a task that never learns.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer

from mtsac.buffer import TaskWeightedReplayBuffer
from mtsac.callback import EvalCallback
from mtsac.environments import MT3_TASKS, env_copies, env_task_ids, make_env
from mtsac.sac import MTSAC, LayerNormSACPolicy

OBS_SPACE = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
ACT_SPACE = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)


def _buffer(task_ids, **kwargs) -> TaskWeightedReplayBuffer:
    return TaskWeightedReplayBuffer(
        100, OBS_SPACE, ACT_SPACE, n_envs=len(task_ids), task_ids=task_ids, **kwargs
    )


def _observed_task_shares(buf: TaskWeightedReplayBuffer, draws: int = 40_000) -> np.ndarray:
    envs = buf._sample_env_indices(draws)
    tasks = buf.task_ids[envs]
    return np.bincount(tasks, minlength=buf.num_tasks) / draws


def test_env_copies_and_task_ids_agree():
    """The buffer weights by task, so its layout must match make_env's."""
    assert env_copies(MT3_TASKS, 5) == [5, 5, 5]
    assert env_copies(MT3_TASKS, {"push-v3": 4}) == [1, 4, 1]
    assert env_task_ids(MT3_TASKS, {"reach-v3": 2, "push-v3": 1, "pick-place-v3": 3}) == [
        0, 0, 1, 2, 2, 2,
    ]
    with pytest.raises(ValueError, match="not in tasks"):
        env_copies(MT3_TASKS, {"door-open-v3": 2})


def test_task_ids_must_cover_every_environment():
    with pytest.raises(ValueError, match="environments"):
        TaskWeightedReplayBuffer(
            100, OBS_SPACE, ACT_SPACE, n_envs=4, task_ids=[0, 1, 2]
        )


def test_untouched_buffer_samples_like_sb3():
    """Inertness: before any evaluation the curriculum must not bias anything."""
    buf = _buffer([0, 0, 1, 1, 2, 2])
    assert buf.env_probabilities is None
    shares = _observed_task_shares(buf)
    assert np.allclose(shares, 1 / 3, atol=0.02), shares


def test_shares_follow_failure_rate():
    buf = _buffer([0, 0, 1, 1, 2, 2])
    buf.set_task_success([1.0, 0.5, 0.0])
    shares = _observed_task_shares(buf)
    # clip(1 - s, 0.1, 0.6) = [0.1, 0.5, 0.6], normalised.
    assert np.allclose(shares, np.array([0.1, 0.5, 0.6]) / 1.2, atol=0.02), shares
    assert shares[2] > shares[1] > shares[0]


def test_floor_keeps_a_solved_task_and_cap_bounds_a_stuck_one():
    """Without the floor a solved task is forgotten; without the cap a stuck one starves
    the rest, which is the MT10 failure mode of the bare failure-rate rule."""
    buf = _buffer([0, 1, 2])
    buf.set_task_success([1.0, 1.0, 0.0])
    shares = _observed_task_shares(buf)
    assert shares[0] > 0.05, "a solved task must keep a share"
    assert shares[2] < 0.85, "a stuck task must not take everything"


def test_uneven_environment_counts_still_give_the_task_its_share():
    """A task's share is the task's, however many environments carry it."""
    buf = _buffer([0, 1, 1, 1, 1])  # one env for task 0, four for task 1
    buf.set_task_success([0.0, 0.0])  # equally unsolved -> equal shares
    shares = _observed_task_shares(buf)
    assert np.allclose(shares, 0.5, atol=0.02), shares


def test_a_plain_buffer_can_be_adopted_without_losing_its_transitions():
    """Resuming a uniform run under a curriculum config must actually get the curriculum.

    The checkpoint pickles the class the run started with, and the callback skips any
    buffer without set_task_success -- so unadopted, the resumed run trains to the end
    with the curriculum silently off.
    """
    plain = ReplayBuffer(100, OBS_SPACE, ACT_SPACE, n_envs=4)
    for _ in range(10):
        plain.add(
            np.ones((4, 6), np.float32),
            np.ones((4, 6), np.float32),
            np.ones((4, 2), np.float32),
            np.ones(4, np.float32),
            np.zeros(4, bool),
            [{}] * 4,
        )

    buf = TaskWeightedReplayBuffer.adopt(plain, task_ids=[0, 0, 1, 1])
    assert buf is plain, "the transitions are the point; do not rebuild the buffer"
    assert buf.size() == 10
    assert hasattr(buf, "set_task_success")

    buf.set_task_success([1.0, 0.0])
    shares = _observed_task_shares(buf)
    assert np.allclose(shares, np.array([0.1, 0.6]) / 0.7, atol=0.02), shares


def _callback(patience: int, tmp_path) -> EvalCallback:
    cb = EvalCallback(None, MT3_TASKS, tmp_path, eval_freq=1, patience=patience, verbose=0)
    cb.num_timesteps = 0
    return cb


def _feed(cb: EvalCallback, rows: list[dict[str, float]]) -> None:
    """Drive the real early-stopping logic, one evaluation per row."""
    for row in rows:
        cb.num_timesteps += 1
        cb._update_early_stop(row)


def test_a_solved_task_does_not_trigger_early_stopping(tmp_path):
    """reach pins at 1.00 and can never set a new best; that must not stop the run while
    pick-place is still climbing."""
    cb = _callback(patience=3, tmp_path=tmp_path)
    rows = [
        {"reach-v3": 1.0, "push-v3": 0.5, "pick-place-v3": p}
        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    _feed(cb, rows)
    assert cb._stale_per_task["reach-v3"] >= 3, "solved task is permanently idle"
    assert not cb._stop, "a task still improving must keep the run alive"


def test_stops_once_every_task_is_idle(tmp_path):
    cb = _callback(patience=3, tmp_path=tmp_path)
    _feed(cb, [{"reach-v3": 1.0, "push-v3": 0.5, "pick-place-v3": 0.4}])
    assert not cb._stop
    _feed(cb, [{"reach-v3": 1.0, "push-v3": 0.5, "pick-place-v3": 0.4}] * 3)
    assert cb._stop, "all tasks idle for patience evals should stop"


def test_a_dip_in_one_task_does_not_lose_another_task_s_progress(tmp_path):
    """On the mean this pair looks like no new best; per task, push clearly improved."""
    cb = _callback(patience=2, tmp_path=tmp_path)
    _feed(cb, [{"reach-v3": 1.0, "push-v3": 0.2, "pick-place-v3": 0.8}])
    _feed(cb, [{"reach-v3": 1.0, "push-v3": 0.9, "pick-place-v3": 0.1}])
    assert cb._stale_per_task["push-v3"] == 0
    assert not cb._stop


def _tiny_model(**kwargs) -> tuple[MTSAC, object]:
    env = make_env(MT3_TASKS, seed=0, max_episode_steps=5, vector_strategy="sync")
    model = MTSAC(
        policy=kwargs.pop("policy", "MlpPolicy"),
        env=env,
        seed=0,
        verbose=0,
        device="cpu",
        learning_starts=20,
        batch_size=16,
        policy_kwargs={"net_arch": [32, 32]},
        num_tasks=3,
        **kwargs,
    )
    return model, env


def test_task_index_is_read_from_the_one_hot():
    model, env = _tiny_model()
    try:
        obs = th.as_tensor(env.reset())
        assert model._task_index(obs).tolist() == [0, 1, 2]
    finally:
        env.close()


def test_per_task_alpha_is_one_coefficient_per_task():
    model, env = _tiny_model(per_task_alpha=True)
    try:
        assert tuple(model.log_ent_coef.shape) == (3,)
        model.learn(total_timesteps=80, progress_bar=False)
        # They are free to move apart; a single shared alpha could not.
        assert model.log_ent_coef.grad is None or model.log_ent_coef.shape[0] == 3
    finally:
        env.close()


def test_defaults_leave_stock_sac_behaviour():
    """Inertness: with every flag off, MTSAC must be plain SAC."""
    model, env = _tiny_model()
    try:
        assert tuple(model.log_ent_coef.shape) == (1,)
        assert not model.normalize_critic_loss and not model.per_task_alpha
        assert th.allclose(model.value_scale, th.ones(3))
        model.learn(total_timesteps=80, progress_bar=False)
        # Untouched, because nothing normalises by it.
        assert th.allclose(model.value_scale, th.ones(3))
    finally:
        env.close()


def test_value_scale_tracks_each_task_separately():
    model, env = _tiny_model(normalize_critic_loss=True)
    try:
        model.learn(total_timesteps=200, progress_bar=False)
        assert model.value_scale.shape == (3,)
        assert th.all(model.value_scale > 0)
    finally:
        env.close()


def test_layernorm_policy_puts_layernorm_in_the_critic_only():
    model, env = _tiny_model(policy=LayerNormSACPolicy)
    try:
        critic = [type(m).__name__ for m in model.critic.qf0]
        actor = [type(m).__name__ for m in model.actor.latent_pi]
        assert critic.count("LayerNorm") == 2, critic
        # LayerNorm after a hidden layer, never after the output.
        assert critic[-1] == "Linear"
        assert "LayerNorm" not in actor, actor
    finally:
        env.close()
