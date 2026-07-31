"""Roll out a policy and measure per-environment success rate and return."""

from __future__ import annotations

import numpy as np


def evaluate(model, env, n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Run ``n_episodes`` per sub-environment with the deterministic policy.

    Returns the success rate and mean return of each sub-environment, in order.
    An episode counts as a success if ``info["success"]`` was set at any point during
    it, which is the Meta-World convention.

    All sub-environments run episodes of the same length and never terminate early, so
    they finish together and one loop covers the whole batch.
    """
    n = env.num_envs
    successes = np.zeros(n)
    returns = np.zeros(n)

    for _ in range(n_episodes):
        obs = env.reset()
        hit = np.zeros(n, dtype=bool)
        dones = np.zeros(n, dtype=bool)
        while not dones.all():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            if env.render_mode is not None:
                env.render()
            returns += rewards
            for i, info in enumerate(infos):
                hit[i] |= float(info.get("success", 0.0)) > 0.0
        successes += hit

    return successes / n_episodes, returns / n_episodes
