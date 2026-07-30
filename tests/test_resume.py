"""A multi-day run must survive an interruption, replay buffer included.

Doubles as the end-to-end smoke test: config -> env -> SAC -> eval -> saved model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import train

TINY = {
    "algo": "sac",
    "id": "tiny",
    "tasks": ["reach-v3"],
    "env": {"max_episode_steps": 10, "use_one_hot": False, "vector_strategy": "sync"},
    "sac": {
        "policy": "MlpPolicy",
        "buffer_size": 500,
        "learning_starts": 8,
        "batch_size": 8,
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [16, 16]},
    },
    "train": {"total_steps": 40, "eval_freq": 1000, "n_eval_episodes": 1, "checkpoint_freq": 20},
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway ROOT, so tests never write into the real results/ directory."""
    (tmp_path / "param").mkdir()
    monkeypatch.setattr(train, "ROOT", tmp_path)
    return tmp_path


def write_config(workspace: Path, **train_overrides) -> None:
    cfg = {**TINY, "train": {**TINY["train"], **train_overrides}}
    (workspace / "param" / "tiny.yaml").write_text(yaml.safe_dump(cfg))


def test_checkpoint_written_then_resumed(workspace, capsys):
    write_config(workspace)
    train.main(["tiny"])

    run_dir = workspace / "results" / "tiny_s0"
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "checkpoint" / "model.zip").exists()
    assert (run_dir / "checkpoint" / "replay_buffer.pkl").exists(), (
        "resume without the replay buffer is not a resume"
    )

    # Raise the budget and resume: it must continue, not restart from zero.
    write_config(workspace, total_steps=80)
    capsys.readouterr()
    train.main(["tiny", "--resume"])
    assert "resumed from" in capsys.readouterr().out


def test_finished_run_is_not_restarted(workspace, capsys):
    write_config(workspace)
    train.main(["tiny"])
    capsys.readouterr()
    train.main(["tiny", "--resume"])
    assert "already complete" in capsys.readouterr().out


def test_resume_refuses_without_replay_buffer(workspace):
    write_config(workspace)
    train.main(["tiny"])
    (workspace / "results" / "tiny_s0" / "checkpoint" / "replay_buffer.pkl").unlink()

    write_config(workspace, total_steps=80)
    with pytest.raises(FileNotFoundError, match="empty replay buffer"):
        train.main(["tiny", "--resume"])
