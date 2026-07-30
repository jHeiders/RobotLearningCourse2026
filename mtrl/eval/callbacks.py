"""Per-task evaluation.

The per-task breakdown is the point: an averaged success rate hides exactly the
inter-task interference the project is meant to analyse.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class MultiTaskEvalCallback(BaseCallback):
    """Roll out the deterministic policy and log success rate per task.

    Success follows the Meta-World convention: an episode counts as a success if
    ``info["success"]`` was ever set during it, not only at the final step.

    ``eval_freq_steps`` is in *total* environment steps summed across tasks, matching
    the x-axis convention used for every learning curve in this project.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq_steps: int,
        n_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq_steps = eval_freq_steps
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.last_results: dict[str, float] = {}
        self._next_eval_at = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval_at:
            self._next_eval_at = self.num_timesteps + self.eval_freq_steps
            self.evaluate()
        return True

    def evaluate(self, n_episodes: int | None = None) -> dict[str, float]:
        n_episodes = self.n_episodes if n_episodes is None else n_episodes
        env = self.eval_env
        n = env.num_envs
        labels = env.get_attr("task_labels")[0]

        successes: list[list[float]] = [[] for _ in range(n)]
        returns: list[list[float]] = [[] for _ in range(n)]
        hit_success = np.zeros(n, dtype=bool)
        running_return = np.zeros(n, dtype=np.float64)

        obs = env.reset()
        while min(len(s) for s in successes) < n_episodes:
            actions, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, infos = env.step(actions)
            running_return += rewards
            for i, info in enumerate(infos):
                if float(info.get("success", 0.0)) > 0.0:
                    hit_success[i] = True
                if dones[i]:
                    successes[i].append(float(hit_success[i]))
                    returns[i].append(float(running_return[i]))
                    hit_success[i] = False
                    running_return[i] = 0.0

        by_task_success: dict[str, list[float]] = defaultdict(list)
        by_task_return: dict[str, list[float]] = defaultdict(list)
        for i, label in enumerate(labels):
            by_task_success[label].extend(successes[i][:n_episodes])
            by_task_return[label].extend(returns[i][:n_episodes])

        results: dict[str, float] = {}
        for task in sorted(by_task_success):
            results[f"eval/success/{task}"] = float(np.mean(by_task_success[task]))
            results[f"eval/return/{task}"] = float(np.mean(by_task_return[task]))

        task_means = [np.mean(v) for v in by_task_success.values()]
        results["eval/success/mean"] = float(np.mean(task_means))
        # Spread across tasks is the stability number the project is graded on.
        results["eval/success/std_across_tasks"] = float(np.std(task_means))
        results["eval/return/mean"] = float(
            np.mean([np.mean(v) for v in by_task_return.values()])
        )

        for key, value in results.items():
            self.logger.record(key, value)
        self.logger.record("eval/steps_per_task", self.num_timesteps / max(len(labels), 1))
        # Flush immediately rather than waiting for SB3's own episode-count cadence.
        # Without this, eval metrics are written at whatever step the next rollout dump
        # happens to land on, and the final evaluation after learn() returns is never
        # written at all — it would reach final_eval.json but never TensorBoard or W&B.
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            summary = "  ".join(
                f"{t}={results[f'eval/success/{t}']:.2f}" for t in sorted(by_task_success)
            )
            print(
                f"[eval @ {self.num_timesteps} steps] "
                f"mean={results['eval/success/mean']:.3f}  {summary}"
            )

        self.last_results = results
        return results
