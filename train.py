"""Train SAC on a Meta-World task set.

    python train.py mt3                     # runs param/mt3.yaml
    python train.py mt3 --seed 1
    python train.py mt3 --wandb
    python train.py mt3 --resume            # continue an interrupted run

Writes results/<id>_s<seed>/ with TensorBoard logs, best_model.zip and final_model.zip.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CallbackList

from mtsac.callback import EvalCallback
from mtsac.checkpoint import LatestCheckpointCallback
from mtsac.config import load_config
from mtsac.environments import make_env, resolve_tasks
from mtsac.sac import ALGO_TABLE

ROOT = Path(__file__).resolve().parent

WANDB_PROJECT = "robot-learning-mtrl"
# The shared team. W&B otherwise falls back to your personal entity, where your partner
# cannot see the run and nothing warns you. $WANDB_ENTITY overrides — but devcontainer.json
# passes it through as "" (not absent) when the host has no such variable, and
# os.environ.get's default only applies to a missing key, not an empty one. `or` catches both.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or "robotlearningcourse2026"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="name of a file in param/, without .yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument(
        "--resume", action="store_true", help="continue from the run's latest checkpoint"
    )
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / "param" / f"{args.config}.yaml")
    tasks = resolve_tasks(cfg["tasks"])
    run_name = f"{cfg['id']}_s{args.seed}"
    run_dir = ROOT / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # The exact settings this run used, with the task set resolved to real ids. param/
    # keeps changing as you tune; this is the copy that says what produced these numbers.
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({**cfg, "tasks": tasks, "seed": args.seed}, sort_keys=False)
    )

    train_env = make_env(tasks, seed=args.seed, **cfg["env"])
    eval_env = make_env(tasks, seed=args.seed, eval_mode=True, **cfg["env"])

    wandb_run = None
    if args.wandb:
        import wandb

        # Seeds of one config collapse into a group, so the dashboard stays readable.
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=cfg["id"],
            tags=[cfg["algo"], cfg["id"], f"seed{args.seed}"],
            config=cfg,
            sync_tensorboard=True,
        )

    algo = ALGO_TABLE[cfg["algo"]]
    ckpt_dir = run_dir / "checkpoint"
    already_done = 0

    if args.resume and (ckpt_dir / "model.zip").exists():
        if not (ckpt_dir / "replay_buffer.pkl").exists():
            # Without the buffer this is not a resume — SAC would carry on with an empty
            # one, which is a different and much worse experiment.
            raise FileNotFoundError(
                f"{ckpt_dir}/model.zip exists but replay_buffer.pkl does not; "
                "refusing to 'resume' from an empty replay buffer"
            )
        model = algo.load(
            ckpt_dir / "model", env=train_env, tensorboard_log=str(run_dir / "tb")
        )
        model.load_replay_buffer(ckpt_dir / "replay_buffer")
        already_done = model.num_timesteps
        print(f"resumed from {ckpt_dir} at {already_done} steps")
    else:
        model = algo(
            env=train_env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            **cfg["sac"],
        )

    remaining = cfg["train"]["total_steps"] - already_done
    if remaining <= 0:
        print(f"{run_name} already complete ({already_done} steps)")
        return

    callbacks = [
        EvalCallback(
            eval_env,
            tasks,
            run_dir,
            eval_freq=cfg["train"]["eval_freq"],
            n_episodes=cfg["train"]["n_eval_episodes"],
        )
    ]
    checkpoint_freq = cfg["train"].get("checkpoint_freq", 0)
    if checkpoint_freq > 0:
        callbacks.append(LatestCheckpointCallback(ckpt_dir, checkpoint_freq))

    try:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList(callbacks),
            log_interval=10,
            tb_log_name="run",
            reset_num_timesteps=already_done == 0,
            progress_bar=True,
        )
        model.save(run_dir / "final_model")
    finally:
        train_env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
