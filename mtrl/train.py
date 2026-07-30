"""Single entrypoint for every experiment.

    uv run python -m mtrl.train --config configs/single_task/reach.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

import mtrl.algos  # noqa: F401  (populates the algo registry)
from mtrl.checkpoint import LatestCheckpointCallback
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
    p.add_argument(
        "--wandb-entity",
        # Defaulted to the shared team on purpose: W&B otherwise falls back to your
        # personal entity, where your partner cannot see the run, and nothing warns you.
        default=os.environ.get("WANDB_ENTITY", "robotlearningcourse2026"),
        help="W&B team (default: the shared team). $WANDB_ENTITY overrides.",
    )
    p.add_argument("--no-eval", action="store_true", help="skip periodic evaluation")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from the run's latest checkpoint if one exists",
    )
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

        # group/job_type/tags are what make the shared dashboard usable: seeds of one
        # config collapse into a group, and the two variant arms of a bake-off sit side
        # by side under the same group with different job types.
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=cfg.name,
            job_type=cfg.variant,
            tags=[cfg.algo.name, cfg.name, cfg.variant, f"seed{args.seed}"],
            config=metadata,
            sync_tensorboard=True,
            save_code=True,
        )

    model = ALGOS.get(cfg.algo.name)(cfg, train_env, args.seed, str(run_dir / "tb"))

    ckpt_dir = run_dir / "checkpoint"
    already_done = 0
    if args.resume and (ckpt_dir / "model.zip").exists():
        model = type(model).load(
            ckpt_dir / "model", env=train_env, tensorboard_log=str(run_dir / "tb")
        )
        if (ckpt_dir / "replay_buffer.pkl").exists():
            model.load_replay_buffer(ckpt_dir / "replay_buffer")
        else:
            # Without the buffer this is not a resume — SAC would carry on with an empty
            # one, which is a different (and much worse) experiment.
            raise FileNotFoundError(
                f"{ckpt_dir}/model.zip exists but replay_buffer.pkl does not; "
                "refusing to 'resume' from an empty replay buffer"
            )
        already_done = model.num_timesteps
        print(f"resumed from {ckpt_dir} at {already_done} steps")

    remaining = cfg.train.total_steps - already_done
    if remaining <= 0:
        print(f"{run_name} already complete ({already_done} steps)")
        return cfg

    callbacks = []
    if cfg.train.checkpoint_freq > 0:
        callbacks.append(LatestCheckpointCallback(ckpt_dir, cfg.train.checkpoint_freq))
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

    eval_cb = next((c for c in callbacks if isinstance(c, MultiTaskEvalCallback)), None)
    try:
        model.learn(
            total_timesteps=remaining,
            log_interval=cfg.train.log_interval,
            callback=CallbackList(callbacks) if callbacks else None,
            tb_log_name="run",
            reset_num_timesteps=already_done == 0,
            progress_bar=False,
        )
        model.save(run_dir / "model")
        if eval_cb is not None:
            # Many more episodes than the curve uses: this is the number that gets
            # compared against the grading thresholds, so it needs the resolution.
            final = eval_cb.evaluate(n_episodes=cfg.eval.final_n_episodes)
            final["n_episodes_per_task"] = cfg.eval.final_n_episodes
            (run_dir / "final_eval.json").write_text(json.dumps(final, indent=2))
    finally:
        train_env.close()
        if eval_cb is not None:
            eval_cb.eval_env.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f"saved -> {run_dir}")
    return cfg


if __name__ == "__main__":
    run(parse_args())
