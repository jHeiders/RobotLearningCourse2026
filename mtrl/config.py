"""Typed run configuration loaded from YAML.

A run is fully identified by (config file, git SHA, seed). Configs are committed;
results are not. ``extends:`` gives one level of inheritance from ``base.yaml``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    # "custom" builds the task set from `tasks`; "MT10"/"MT50" use the official
    # Meta-World benchmark ids and ignore `tasks`.
    benchmark: str = "custom"
    tasks: list[str] = field(default_factory=lambda: ["reach-v3"])
    # None -> automatic: one-hot iff there is more than one task.
    use_one_hot: bool | None = None
    # >1 only makes sense for single-task runs, where it buys throughput.
    envs_per_task: int = 1
    max_episode_steps: int = 500
    reward_function_version: str = "v2"
    terminate_on_success: bool = False
    task_select: str = "random"
    vector_strategy: str = "async"
    # Reward/observation shaping — the single-task variant axis. None/False is the
    # standard benchmark setting that the grading thresholds assume.
    reward_normalization_method: str | None = None
    normalize_observations: bool = False

    # Note: the number of parametric goal variations per task is fixed at Meta-World's
    # default of 50. Its `num_goals` kwarg only reaches the MT1/MT10/MT50 entry points,
    # not `custom-mt-envs`, so exposing it would silently desynchronise MT3 from MT10.
    # `num_tasks` is *not* that knob — it is the one-hot width, derived automatically.


@dataclass
class AlgoConfig:
    name: str = "sac"
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 4_000
    batch_size: int = 512
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    # -1 means "as many gradient steps as env steps collected", i.e. UTD = 1,
    # which is the MT-SAC convention.
    gradient_steps: int = -1
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [400, 400, 400])


@dataclass
class EvalConfig:
    # freq is in *total* environment steps summed across tasks.
    freq: int = 100_000
    n_episodes: int = 5
    # The reported result needs finer resolution than the curve does. With 5 episodes a
    # success rate can only land on multiples of 0.2, so "34 %" is not even expressible
    # — and the grading thresholds are 30 % and 40 %. The final evaluation therefore
    # uses many more episodes; it runs once, so it is cheap.
    final_n_episodes: int = 50
    deterministic: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000
    log_interval: int = 10
    # Total env steps between checkpoints. 0 disables checkpointing.
    checkpoint_freq: int = 200_000


@dataclass
class Config:
    name: str = "unnamed"
    variant: str = "base"
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def run_name(self, seed: int) -> str:
        return f"{self.algo.name}_{self.name}_{self.variant}_s{seed}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:8]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_raw((path.parent / parent).resolve()), raw)


def _build(cls: type, data: dict) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    raw = _load_raw(Path(path).resolve())
    return Config(
        name=raw.get("name", "unnamed"),
        variant=raw.get("variant", "base"),
        env=_build(EnvConfig, raw.get("env", {})),
        algo=_build(AlgoConfig, raw.get("algo", {})),
        eval=_build(EvalConfig, raw.get("eval", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )


def git_sha() -> str:
    """Short commit hash, or 'nogit' outside a repository. Logged with every run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"
