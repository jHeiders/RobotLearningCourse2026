"""Adapt a Gymnasium vector environment to the Stable-Baselines3 ``VecEnv`` API."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from stable_baselines3.common.vec_env import VecEnv


def split_infos(infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """Gymnasium's batched info dict ``{key: array, "_key": mask}`` -> one dict per env."""
    out: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        nested = split_infos(value, num_envs) if isinstance(value, dict) else None
        for i in range(num_envs):
            if mask is None or mask[i]:
                out[i][key] = value[i] if nested is None else nested[i]
    return out


class GymVecToSB3(VecEnv):
    """Make a Meta-World vector environment usable by SB3.

    Meta-World already autoresets in the same step, which is what SB3 expects. What
    still has to be translated:

    * the batched info dict, into one dict per environment;
    * the terminal observation, which SB3 reads as ``info["terminal_observation"]``;
    * ``info["TimeLimit.truncated"]``, without which SB3 stops bootstrapping the value
      of an episode that ended on the time limit. Meta-World episodes nearly always end
      that way, so omitting it corrupts SAC's targets on almost every episode;
    * ``success``, which Meta-World reports only inside ``final_info`` on the last step.
    """

    def __init__(self, venv: VectorEnv) -> None:
        self.venv = venv
        obs_space = venv.single_observation_space
        super().__init__(
            num_envs=venv.num_envs,
            # Meta-World declares float64 bounds that its own observations violate, which
            # makes Gymnasium's checker warn on every step. SAC's MLP never reads the
            # bounds, and float32 halves replay-buffer memory.
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_space.shape, dtype=np.float32
            ),
            action_space=venv.single_action_space,
        )

    @staticmethod
    def _obs(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def reset(self):
        obs, _ = self.venv.reset()
        return self._obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminations, truncations, infos = self.venv.step(self._actions)
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        dones = np.logical_or(terminations, truncations)

        per_env = split_infos(infos, self.num_envs)
        final_obs = infos.get("final_obs")
        for i in np.nonzero(dones)[0]:
            info = per_env[i]
            nested = info.pop("final_info", None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    info.setdefault(key, value)
            info.pop("final_obs", None)
            info["TimeLimit.truncated"] = bool(truncations[i]) and not bool(terminations[i])
            if final_obs is not None:
                info["terminal_observation"] = self._obs(final_obs[i])

        return self._obs(obs), np.asarray(rewards, dtype=np.float32), dones, per_env

    def close(self) -> None:
        self.venv.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        if hasattr(self, attr_name):
            values = [getattr(self, attr_name)] * self.num_envs
        else:
            values = list(self.venv.get_attr(attr_name))
        return [values[i] for i in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        chosen = set(self._get_indices(indices))
        current = list(self.venv.get_attr(attr_name))
        self.venv.set_attr(
            attr_name, [value if i in chosen else current[i] for i in range(self.num_envs)]
        )

    def env_method(self, method_name: str, *args, indices: Any = None, **kwargs) -> list[Any]:
        results = list(self.venv.call(method_name, *args, **kwargs))
        return [results[i] for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Any = None) -> list[bool]:
        # Sub-environments come from Meta-World, not from a stack of SB3 wrappers.
        return [False] * len(list(self._get_indices(indices)))
