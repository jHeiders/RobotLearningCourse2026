"""Print the environment fingerprint. Run this first when two machines disagree.

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "no GPU reported"
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not available (CPU-only is fine — the bottleneck is MuJoCo on CPU)"


def main() -> None:
    import gymnasium
    import metaworld  # noqa: F401  (import proves it loads)
    import numpy
    import stable_baselines3
    import torch

    from mtrl.config import git_sha

    rows = [
        ("platform", f"{platform.system()} {platform.release()}"),
        ("python", sys.version.split()[0]),
        ("numpy", numpy.__version__),
        ("torch", torch.__version__),
        ("torch.cuda", f"available={torch.cuda.is_available()} version={torch.version.cuda}"),
        ("gymnasium", gymnasium.__version__),
        # metaworld does not expose __version__.
        ("metaworld", importlib.metadata.version("metaworld")),
        ("stable_baselines3", stable_baselines3.__version__),
        ("gpu", _gpu()),
        ("git_sha", git_sha()),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
