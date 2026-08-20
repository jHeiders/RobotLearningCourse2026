"""How much reward each MT10 task exposes to a policy that has not learned it yet.

    python probes/peak_reward.py

For each task we roll out a uniform-random policy, record the highest single-step reward
within each episode, and average those over episodes and seeds. Averaging the per-episode
maximum rather than taking a maximum over all episodes matters: the latter is a max
statistic, grows with the episode count and varies by more than 2x between seeds, while
the quantity below is stable to the second decimal.

Runs on CPU with no trained policy, so it is cheap to reproduce.
"""

from __future__ import annotations

import numpy as np

from mtsac.environments import MT10_TASKS, make_env

EPISODES = 50
HORIZON = 200
SEEDS = (0, 1, 2)


def reachable_reward(task: str, seed: int) -> float:
    """Mean over episodes of the largest per-step reward a random policy sees."""
    venv = make_env([task], envs_per_task=1, seed=seed, eval_mode=True)
    rng = np.random.default_rng(seed)
    width = venv.action_space.shape[-1]
    per_episode = []
    try:
        for _ in range(EPISODES):
            venv.reset()
            best = 0.0
            for _ in range(HORIZON):
                action = rng.uniform(-1, 1, size=(1, width)).astype(np.float32)
                _, reward, _, _ = venv.step(action)
                best = max(best, float(np.max(reward)))
            per_episode.append(best)
    finally:
        venv.close()
    return float(np.mean(per_episode))


if __name__ == "__main__":
    print(f"{'task':24}{'reachable reward':>18}")
    for task in MT10_TASKS:
        values = [reachable_reward(task, seed) for seed in SEEDS]
        print(f"{task.replace('-v3', ''):24}{np.mean(values):>18.2f}", flush=True)
