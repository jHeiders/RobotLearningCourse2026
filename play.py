"""Score a trained model, optionally watching it.

    python play.py mt3                      # results/mt3_s0/best_model.zip
    python play.py mt3 --seed 1 --episodes 50
    python play.py st_reach --render
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mtsac.config import load_config
from mtsac.environments import make_env, resolve_tasks
from mtsac.eval import evaluate
from mtsac.sac import ALGO_TABLE

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="name of a file in param/, without .yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="best_model", help="best_model | final_model")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per task")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / "param" / f"{args.config}.yaml")
    tasks = resolve_tasks(cfg["tasks"])
    run_dir = ROOT / "results" / f"{cfg['id']}_s{args.seed}"

    env = make_env(
        tasks,
        seed=args.seed,
        eval_mode=True,
        render_mode="human" if args.render else None,
        **cfg["env"],
    )
    model = ALGO_TABLE[cfg["algo"]].load(
        run_dir / args.model,
        env=env,
        # The checkpoint carries the training buffer's settings, whose task_ids are sized
        # for the training layout and do not fit the one-environment-per-task eval layout.
        # Playing never samples, so the buffer is replaced with an empty default one.
        custom_objects={
            "replay_buffer_class": None,
            "replay_buffer_kwargs": {},
            "buffer_size": 1,
        },
    )

    try:
        successes, returns = evaluate(model, env, args.episodes)
    finally:
        env.close()

    print(f"{run_dir / args.model}  ({args.episodes} episodes per task)")
    for task, success, ret in zip(tasks, successes, returns, strict=True):
        print(f"  {task:<26} success={success:.2f}  return={ret:8.1f}")
    print(f"  {'mean':<26} success={successes.mean():.2f}  return={returns.mean():8.1f}")


if __name__ == "__main__":
    main()
