"""End-to-end smoke test: the training entrypoint must actually run.

Deliberately tiny — this checks the wiring (config -> envs -> SAC -> eval -> artefacts),
not learning. CI runs it on every push so `main` never carries a broken pipeline.
"""

from __future__ import annotations

import argparse
import json

import yaml

from mtrl.config import load_config
from mtrl.train import run


def _tiny_config(tmp_path, **env_overrides) -> str:
    cfg = {
        "name": "smoke",
        "env": {
            "vector_strategy": "sync",
            "max_episode_steps": 10,
            **env_overrides,
        },
        "algo": {
            "learning_starts": 8,
            "batch_size": 8,
            "buffer_size": 500,
            "net_arch": [16, 16],
            "gradient_steps": 1,
        },
        "eval": {"freq": 40, "n_episodes": 1},
        "train": {"total_steps": 60},
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _args(config: str, out, seed: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        seed=seed,
        total_steps=None,
        out=out,
        wandb=False,
        wandb_project="unused",
        no_eval=False,
    )


def test_single_task_train_produces_artifacts(tmp_path):
    cfg_path = _tiny_config(tmp_path, tasks=["reach-v3"])
    run(_args(cfg_path, tmp_path / "results"))

    run_dir = tmp_path / "results" / "sac_smoke_base_s0"
    assert (run_dir / "model.zip").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["seed"] == 0
    assert metadata["tasks"] == ["reach-v3"]
    assert metadata["config_hash"]

    final = json.loads((run_dir / "final_eval.json").read_text())
    assert 0.0 <= final["eval/success/reach-v3"] <= 1.0
    assert "eval/success/mean" in final


def test_multi_task_train_logs_every_task(tmp_path):
    cfg_path = _tiny_config(tmp_path, tasks=["reach-v3", "push-v3", "pick-place-v3"])
    run(_args(cfg_path, tmp_path / "results"))

    final = json.loads(
        (tmp_path / "results" / "sac_smoke_base_s0" / "final_eval.json").read_text()
    )
    for task in ("reach-v3", "push-v3", "pick-place-v3"):
        assert f"eval/success/{task}" in final, "per-task breakdown is the interference metric"
    assert "eval/success/std_across_tasks" in final


def test_config_inheritance_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"name": "b", "algo": {"batch_size": 512, "gamma": 0.99}}))
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump({"extends": "base.yaml", "name": "c", "algo": {"batch_size": 128}})
    )

    cfg = load_config(child)
    assert cfg.name == "c"
    assert cfg.algo.batch_size == 128  # overridden
    assert cfg.algo.gamma == 0.99  # inherited
    assert cfg.run_name(3) == "sac_c_base_s3"


def test_shipped_configs_all_load():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    configs = sorted(root.rglob("*.yaml"))
    assert len(configs) >= 6
    for path in configs:
        cfg = load_config(path)
        assert cfg.algo.name in ("sac",), f"{path.name} selects an unregistered algo"
