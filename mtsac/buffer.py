"""A replay buffer that can bias sampling towards the tasks that are still failing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class TaskWeightedReplayBuffer(ReplayBuffer):
    """Sample transitions per task instead of uniformly over sub-environments.

    An even split of environments across tasks spends the same optimization on a task that
    was solved in the first 100k steps as on one that is still at zero. Measured on MT3:
    ten times the optimization with an even split still left pick-place at exactly 0.00,
    while re-slicing the share fixed it. So the share is what binds, and it is set here
    rather than by hand.

    Each sub-environment runs a fixed task, so a batch's task mix is decided entirely by
    which sub-environments its samples are drawn from -- that is the one line this class
    changes. Weights come from evaluation, via ``set_task_success``, so nothing per-task is
    configured.

    ``floor`` and ``cap`` bound the share. Without a floor, a task's share collapses to
    zero once it is solved and the policy forgets it. Without a cap, a task that never
    improves takes the whole budget -- harmless on MT3, but MT10 contains tasks that sit at
    zero for most published methods, which would starve everything else.
    """

    def __init__(
        self,
        *args,
        task_ids: Sequence[int],
        floor: float = 0.1,
        cap: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._configure(task_ids, floor, cap)

    def _configure(self, task_ids: Sequence[int], floor: float, cap: float) -> None:
        self.task_ids = np.asarray(task_ids, dtype=np.int64)
        if self.task_ids.shape != (self.n_envs,):
            raise ValueError(
                f"task_ids has {self.task_ids.shape[0]} entries but the buffer has "
                f"{self.n_envs} environments"
            )
        self.num_tasks = int(self.task_ids.max()) + 1
        self.floor = floor
        self.cap = cap
        # Until the first evaluation there is nothing to go on, so sample as SB3 does.
        self.env_probabilities: np.ndarray | None = None

    @classmethod
    def adopt(
        cls,
        buffer: ReplayBuffer,
        task_ids: Sequence[int],
        floor: float = 0.1,
        cap: float = 0.6,
    ) -> TaskWeightedReplayBuffer:
        """Take over a plain buffer loaded from a checkpoint, keeping its transitions.

        A checkpoint pickles whichever buffer class the run was started with, so resuming a
        uniform run under a curriculum config loads a plain ``ReplayBuffer`` -- and the
        callback skips any buffer without ``set_task_success``, so the curriculum would
        silently never engage. Only the sampling differs between the two classes; the stored
        arrays are identical, so the loaded buffer is re-classed rather than refilled.
        """
        buffer.__class__ = cls
        buffer._configure(task_ids, floor, cap)
        return buffer  # type: ignore[return-value]

    def set_task_success(self, success: Sequence[float]) -> np.ndarray:
        """Turn per-task success rates into per-sub-environment sampling probabilities."""
        weights = np.clip(1.0 - np.asarray(success, dtype=np.float64), self.floor, self.cap)
        probabilities = np.zeros(self.n_envs, dtype=np.float64)
        for task in range(self.num_tasks):
            envs = np.flatnonzero(self.task_ids == task)
            # Split a task's share evenly over the environments running it.
            probabilities[envs] = weights[task] / len(envs)
        self.env_probabilities = probabilities / probabilities.sum()
        return self.env_probabilities

    def _sample_env_indices(self, size: int) -> np.ndarray:
        if self.env_probabilities is None:
            return np.random.randint(0, high=self.n_envs, size=(size,))
        return np.random.choice(self.n_envs, size=size, p=self.env_probabilities)

    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> ReplayBufferSamples:
        # Mirrors SB3's ReplayBuffer._get_samples; only the env_indices draw differs.
        env_indices = self._sample_env_indices(len(batch_inds))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
