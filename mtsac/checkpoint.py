"""Periodic checkpointing, so a long run survives an interruption.

MT10 is roughly three days per seed on this hardware, on a machine someone also uses.
Losing hour 60 to a reboot, an OOM or a stray Ctrl-C is a schedule-level event — and the
replay buffer has to be saved with the model, or a "resume" silently restarts SAC from an
empty buffer and the run is not the experiment it claims to be.

Only the latest checkpoint is kept: the replay buffer runs to hundreds of megabytes, so
keeping every one of them would fill the disk.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class LatestCheckpointCallback(BaseCallback):
    def __init__(self, path: Path, every_steps: int, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.path = Path(path)
        self.every_steps = every_steps
        self._next_at = every_steps

    def _on_step(self) -> bool:
        if self.every_steps > 0 and self.num_timesteps >= self._next_at:
            self._next_at = self.num_timesteps + self.every_steps
            self.save()
        return True

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name and rename, so a process killed mid-write leaves the
        # previous checkpoint intact rather than a truncated one. The temporary names
        # carry no dot-suffix on purpose: SB3 only appends .zip/.pkl when the path has
        # none, so "model.tmp" would be saved verbatim and the rename would miss it.
        self.model.save(self.path / "model_tmp")
        (self.path / "model_tmp.zip").replace(self.path / "model.zip")
        self.model.save_replay_buffer(self.path / "replay_buffer_tmp")
        (self.path / "replay_buffer_tmp.pkl").replace(self.path / "replay_buffer.pkl")
        if self.verbose:
            print(f"[checkpoint @ {self.num_timesteps} steps] -> {self.path}")
