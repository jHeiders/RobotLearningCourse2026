"""Periodic evaluation during training: log per-task success, keep the best model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mtsac.eval import evaluate


class EvalCallback(BaseCallback):
    """Evaluate every ``eval_freq`` environment steps.

    The per-task breakdown is the point: an averaged success rate hides exactly the
    inter-task interference this project is about. ``eval_freq`` counts total steps
    summed over all environments, which is also the x-axis of every learning curve.
    """

    def __init__(
        self,
        eval_env,
        tasks: list[str],
        run_dir: Path,
        eval_freq: int,
        n_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tasks = list(tasks)
        self.run_dir = Path(run_dir)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.best_success = -1.0
        self._next_eval = 0
        # Several environments may run the same task; report one number per task.
        self._by_task: dict[str, list[int]] = {}
        for i, task in enumerate(self.tasks):
            self._by_task.setdefault(task, []).append(i)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval = self.num_timesteps + self.eval_freq
            self.run_eval()
        return True

    def run_eval(self) -> dict[str, float]:
        successes, returns = evaluate(self.model, self.eval_env, self.n_episodes)
        per_task = {task: float(successes[i].mean()) for task, i in self._by_task.items()}
        mean = float(np.mean(list(per_task.values())))

        for task, idx in self._by_task.items():
            self.logger.record(f"eval/success/{task}", per_task[task])
            self.logger.record(f"eval/return/{task}", float(returns[idx].mean()))
        self.logger.record("eval/success/mean", mean)
        self.logger.record("eval/return/mean", float(returns.mean()))
        # Dump now, so the numbers land on the step they were measured at instead of
        # whenever the next rollout happens to be written.
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            detail = "  ".join(f"{task}={rate:.2f}" for task, rate in per_task.items())
            print(f"[eval @ {self.num_timesteps} steps] mean={mean:.3f}  {detail}")

        if mean > self.best_success:
            self.best_success = mean
            self.model.save(self.run_dir / "best_model")

        return per_task
