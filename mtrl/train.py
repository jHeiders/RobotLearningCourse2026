"""Single entrypoint for every experiment.

    uv run python -m mtrl.train --config configs/single_task/reach.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

import mtrl.algos  # noqa: F401  (populates the algo registry)
from mtrl.config import Config, git_sha, load_config
from mtrl.envs.make import make_vec_env, task_list
from mtrl.eval.callbacks import MultiTaskEvalCallback
from mtrl.registry import ALGOS

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=None, help="override train.total_steps")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="robot-learning-mtrl")
    p.add_argument("--no-eval", action="store_true", help="skip periodic evaluation")
    return p.parse_args()


def run(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.total_steps is not None:
        cfg.train.total_steps = args.total_steps

    run_name = cfg.run_name(args.seed)
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).resolve()
    metadata = {
        "run_name": run_name,
        "config_path": str(
            config_path.relative_to(REPO_ROOT)
            if config_path.is_relative_to(REPO_ROOT)
            else config_path
        ),
        "config_hash": cfg.hash(),
        "git_sha": git_sha(),
        "seed": args.seed,
        "tasks": task_list(cfg.env),
        "config": cfg.to_dict(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    set_random_seed(args.seed)
    train_env = make_vec_env(cfg.env, seed=args.seed, eval_mode=False)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=metadata,
            sync_tensorboard=True,
            save_code=True,
        )

    model = ALGOS.get(cfg.algo.name)(cfg, train_env, args.seed, str(run_dir / "tb"))

    callbacks = []
    if not args.no_eval:
        # env_seed() gives evaluation its own non-overlapping seed window, so eval
        # samples different goal variations than training does.
        eval_env = make_vec_env(cfg.env, seed=args.seed, eval_mode=True)
        callbacks.append(
            MultiTaskEvalCallback(
                eval_env,
                eval_freq_steps=cfg.eval.freq,
                n_episodes=cfg.eval.n_episodes,
                deterministic=cfg.eval.deterministic,
            )
        )

    try:
        model.learn(
            total_timesteps=cfg.train.total_steps,
            log_interval=cfg.train.log_interval,
            callback=CallbackList(callbacks) if callbacks else None,
            tb_log_name="run",
            progress_bar=False,
        )
        model.save(run_dir / "model")
        if callbacks:
            final = callbacks[0].evaluate()
            (run_dir / "final_eval.json").write_text(json.dumps(final, indent=2))
    finally:
        train_env.close()
        if callbacks:
            callbacks[0].eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"saved -> {run_dir}")
    return cfg


if __name__ == "__main__":
    run(parse_args())
