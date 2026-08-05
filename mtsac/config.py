"""Load a ``param/*.yaml`` and reject typos before a multi-day run starts.

The configs are plain dictionaries passed straight through to Meta-World and SB3, which
means an unrecognised key would otherwise be ignored in silence: write ``batchsize:``
instead of ``batch_size:`` and training runs happily on the default. The allowed keys
are read off the functions the values are handed to, so this cannot drift out of date.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml

from mtsac.environments import make_env
from mtsac.sac import ALGO_TABLE

TOP_LEVEL = {"algo", "id", "tasks", "env", "sac", "train"}
TRAIN_KEYS = {"total_steps", "eval_freq", "n_eval_episodes", "checkpoint_freq", "patience"}
# `make_env` arguments that the run supplies itself, not the config.
ENV_ARGS_SET_BY_CODE = {"tasks", "seed", "eval_mode", "render_mode"}


def _reject_unknown(section: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} key(s) {unknown}; allowed: {sorted(allowed)}")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())

    missing = sorted(TOP_LEVEL - set(cfg))
    if missing:
        raise ValueError(f"{path}: missing top-level key(s) {missing}")
    _reject_unknown(cfg, TOP_LEVEL, "top-level")

    if cfg["algo"] not in ALGO_TABLE:
        raise ValueError(f"unknown algo {cfg['algo']!r}; available: {sorted(ALGO_TABLE)}")

    env_keys = set(inspect.signature(make_env).parameters) - ENV_ARGS_SET_BY_CODE
    _reject_unknown(cfg["env"], env_keys, "env")
    _reject_unknown(cfg["train"], TRAIN_KEYS, "train")
    # Validated against the selected algorithm, so a variant's extra arguments are fine.
    _reject_unknown(cfg["sac"], set(inspect.signature(ALGO_TABLE[cfg["algo"]]).parameters), "sac")

    return cfg
