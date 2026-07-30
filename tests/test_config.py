"""Config validation: a typo must fail now, not silently after a day of training."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mtsac.config import load_config

PARAM_DIR = Path(__file__).resolve().parent.parent / "param"


def write(tmp_path: Path, drop: str | None = None, **overrides) -> Path:
    cfg = {
        "algo": "sac",
        "id": "unit",
        "tasks": ["reach-v3"],
        "env": {"max_episode_steps": 10},
        "sac": {"policy": "MlpPolicy", "batch_size": 8},
        "train": {"total_steps": 10, "eval_freq": 10, "n_eval_episodes": 1},
    }
    cfg.update(overrides)
    cfg.pop(drop, None)
    path = tmp_path / "unit.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_valid_config_loads(tmp_path):
    assert load_config(write(tmp_path))["id"] == "unit"


def test_shipped_configs_all_load():
    configs = sorted(PARAM_DIR.glob("*.yaml"))
    assert len(configs) == 5
    for path in configs:
        load_config(path)


def test_misspelled_sac_key_is_rejected(tmp_path):
    """The failure this exists for: `batchsize` would otherwise train on the default."""
    with pytest.raises(ValueError, match="unknown sac key"):
        load_config(write(tmp_path, sac={"policy": "MlpPolicy", "batchsize": 8}))


def test_misspelled_env_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown env key"):
        load_config(write(tmp_path, env={"max_episode_step": 10}))


def test_misspelled_train_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown train key"):
        load_config(write(tmp_path, train={"total_step": 10}))


def test_missing_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing top-level"):
        load_config(write(tmp_path, drop="sac"))


def test_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(write(tmp_path, tasks_=["reach-v3"]))


def test_unknown_algo_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown algo"):
        load_config(write(tmp_path, algo="ppo"))
