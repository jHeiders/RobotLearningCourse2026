"""Periodic evaluation during training: log per-task success, keep the best model."""

from __future__ import annotations

from collections import deque
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
        patience: int | None = None,
        curriculum_window: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tasks = list(tasks)
        self.run_dir = Path(run_dir)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.patience = patience
        self.best_success = -1.0
        self._next_eval = 0
        self._stop = False
        # Early stopping is judged per task, not on the mean. A task that is already at
        # 1.00 can never set a new best, so on the mean it looks permanently stalled and
        # drags the average's high-water mark with it; meanwhile the mean can also miss a
        # new best when one task improves and another dips. Counting per task, a run only
        # stops once *every* task has gone quiet.
        self._best_per_task: dict[str, float] = dict.fromkeys(self._task_names(tasks), -1.0)
        self._stale_per_task: dict[str, int] = dict.fromkeys(self._task_names(tasks), 0)
        # A single evaluation is noisy enough to swing a task's share; averaging the last
        # few keeps the curriculum from chasing that noise.
        self._recent_success: deque[list[float]] = deque(maxlen=curriculum_window)
        # Several environments may run the same task; report one number per task.
        self._by_task: dict[str, list[int]] = {}
        for i, task in enumerate(self.tasks):
            self._by_task.setdefault(task, []).append(i)

    @staticmethod
    def _task_names(tasks: list[str]) -> list[str]:
        """Distinct tasks, in first-appearance order."""
        return list(dict.fromkeys(tasks))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval = self.num_timesteps + self.eval_freq
            self.run_eval()
        return not self._stop

    def _update_curriculum(self, per_task: dict[str, float]) -> None:
        """Hand the measured per-task success to a buffer that samples by task.

        Does nothing for the plain buffer, so the same callback serves every run. The task
        order is ``self.tasks``, which is also the order ``env_task_ids`` numbers them in.
        """
        buffer = getattr(self.model, "replay_buffer", None)
        if not hasattr(buffer, "set_task_success"):
            return
        self._recent_success.append([per_task[task] for task in self._by_task])
        smoothed = np.mean(self._recent_success, axis=0)
        probabilities = buffer.set_task_success(smoothed)
        for i, task in enumerate(self._by_task):
            share = float(probabilities[buffer.task_ids == i].sum())
            self.logger.record(f"curriculum/share/{task}", share)

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

        self._update_curriculum(per_task)

        # The saved model is still the best mean, which is the number the run is judged on.
        if mean > self.best_success:
            self.best_success = mean
            self.model.save(self.run_dir / "best_model")

        self._update_early_stop(per_task)

        return per_task

    def _update_early_stop(self, per_task: dict[str, float]) -> bool:
        """Count how long each task has gone without a personal best; stop when all have.

        Returns whether the run should stop, so this is also what the tests exercise.
        """
        for task, rate in per_task.items():
            if rate > self._best_per_task[task]:
                self._best_per_task[task] = rate
                self._stale_per_task[task] = 0
            else:
                self._stale_per_task[task] += 1

        if self.patience is not None and min(self._stale_per_task.values()) >= self.patience:
            self._stop = True
            if self.verbose:
                stale = "  ".join(f"{t}={n}" for t, n in self._stale_per_task.items())
                print(
                    f"[eval @ {self.num_timesteps} steps] every task idle for "
                    f"{self.patience}+ evals ({stale}), stopping early"
                )
        return self._stop
