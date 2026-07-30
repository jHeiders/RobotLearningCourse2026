"""Adapt a Gymnasium 1.x ``VectorEnv`` to the Stable-Baselines3 ``VecEnv`` contract.

Meta-World's vector entry points already default to ``AutoresetMode.SAME_STEP``, which
matches what SB3 expects: ``step`` returns the *reset* observation and carries the
terminal one in the info. So the feared autoreset mismatch does not arise. What still
has to be translated:

* Gymnasium batches infos as ``{key: array, "_key": mask}``; SB3 wants a list of
  per-env dicts.
* The terminal observation lives in ``infos["final_obs"][i]``; SB3 reads
  ``infos[i]["terminal_observation"]``.
* SB3 needs ``infos[i]["TimeLimit.truncated"]`` to know it must still bootstrap the
  value of an episode that ended on the time limit. Meta-World episodes nearly always
  end that way (500-step truncation with ``terminated=False``), so getting this wrong
  would silently corrupt SAC's value targets on essentially every episode.
* On the terminal step Meta-World reports ``success`` only inside ``final_info``. We
  lift those keys up so ``info["success"]`` is present on every step.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn


def _unbounded_float32(space: gym.spaces.Box) -> gym.spaces.Box:
    """Re-declare an observation space as unbounded float32.

    Meta-World declares float64 bounds that its own observations violate (the goal
    dimensions are declared ``[0, 0]`` but populated at runtime), which makes the
    Gymnasium passive checker warn on every step. SAC's MLP policy never consults the
    bounds, so declaring them unbounded is behaviourally free, and float32 halves
    replay-buffer memory.
    """
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32)


def split_infos(infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """Gymnasium's batched info dict -> SB3's list of per-env info dicts."""
    out: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if isinstance(value, dict):
            nested = split_infos(value, num_envs)
            for i in range(num_envs):
                if mask is None or mask[i]:
                    out[i][key] = nested[i]
            continue
        for i in range(num_envs):
            if mask is None or mask[i]:
                out[i][key] = value[i]
    return out


class GymVecToSB3(VecEnv):
    """Wrap a Gymnasium ``VectorEnv`` so SB3 algorithms can train on it.

    ``task_labels`` names the task running in each sub-environment, in order. It is
    what per-task success logging keys off, so it must line up with the one-hot block
    appended to the observation.
    """

    def __init__(self, venv: VectorEnv, task_labels: list[str]) -> None:
        if len(task_labels) != venv.num_envs:
            raise ValueError(
                f"got {len(task_labels)} task labels for {venv.num_envs} environments"
            )
        self.venv = venv
        self.task_labels = list(task_labels)
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=_unbounded_float32(venv.single_observation_space),
            action_space=venv.single_action_space,
        )
        self.render_mode = getattr(venv, "render_mode", None)
        self._actions: np.ndarray | None = None

    @staticmethod
    def _obs(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def reset(self) -> VecEnvObs:
        if self._seeds is not None:
            obs, _ = self.venv.reset(seed=self._seeds)
            self._reset_seeds()
        else:
            obs, _ = self.venv.reset()
        return self._obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terminations, truncations, infos = self.venv.step(self._actions)
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        dones = np.logical_or(terminations, truncations)

        per_env = split_infos(infos, self.num_envs)
        final_obs = infos.get("final_obs")

        for i in np.nonzero(dones)[0]:
            info = per_env[i]
            # Meta-World reports the episode's last `success` inside final_info.
            nested = info.pop("final_info", None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    info.setdefault(key, value)
            info.pop("final_obs", None)
            info["TimeLimit.truncated"] = bool(truncations[i]) and not bool(terminations[i])
            if final_obs is not None:
                info["terminal_observation"] = self._obs(final_obs[i])

        for i, info in enumerate(per_env):
            info["task"] = self.task_labels[i]
            info["task_idx"] = i

        return (
            self._obs(obs),
            np.asarray(rewards, dtype=np.float32),
            dones,
            per_env,
        )

    def close(self) -> None:
        self.venv.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        idx = list(self._get_indices(indices))
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)] * len(idx)
        values = list(self.venv.get_attr(attr_name))
        return [values[i] for i in idx]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        idx = set(self._get_indices(indices))
        current = list(self.venv.get_attr(attr_name))
        self.venv.set_attr(
            attr_name, [value if i in idx else current[i] for i in range(self.num_envs)]
        )

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        idx = list(self._get_indices(indices))
        results = list(self.venv.call(method_name, *args, **kwargs))
        return [results[i] for i in idx]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Any = None) -> list[bool]:
        # Sub-environments are built by Meta-World, not stacked with SB3 wrappers.
        return [False] * len(list(self._get_indices(indices)))

    def get_images(self) -> list[np.ndarray | None]:
        return list(self.venv.call("render"))
