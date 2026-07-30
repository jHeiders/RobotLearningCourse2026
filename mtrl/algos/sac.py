"""SAC baseline.

Multi-task SAC differs from this only in what it is handed: a task set instead of a
single task, a one-hot appended to the observation, and one shared replay buffer across
all of them. That makes plain SB3 SAC a legitimate MT-SAC with a shared backbone, and
the natural starting point before any architectural variant.
"""

from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

from mtrl.config import Config
from mtrl.registry import ALGOS


@ALGOS.register("sac")
def build_sac(cfg: Config, env: VecEnv, seed: int, tensorboard_log: str | None) -> SAC:
    algo = cfg.algo
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=algo.learning_rate,
        buffer_size=algo.buffer_size,
        learning_starts=algo.learning_starts,
        batch_size=algo.batch_size,
        tau=algo.tau,
        gamma=algo.gamma,
        train_freq=algo.train_freq,
        gradient_steps=algo.gradient_steps,
        ent_coef=algo.ent_coef,
        policy_kwargs={"net_arch": list(algo.net_arch)},
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
