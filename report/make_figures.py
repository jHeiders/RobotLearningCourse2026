"""Regenerate the report's figures from the TensorBoard logs in results/.

    python report/make_figures.py

Reads every event file of a run in modification order, so a resumed run's re-logged
steps replace the originals rather than appearing twice. Writes PDF into report/fig/.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "fig"

ARMS = [
    ("mt10_s0", "uniform", "tab:gray", "-"),
    ("mt10_curr_s0", "+ curriculum (warm start)", "tab:orange", "--"),
    ("mt10_norm_curr_s0", "+ curriculum + normalisation", "tab:blue", "-"),
]


def scalars(run: str, prefix: str) -> dict[str, dict[int, float]]:
    """tag suffix -> {step: value}, later event files winning on duplicate steps."""
    out: dict[str, dict[int, float]] = {}
    files = sorted(glob.glob(str(ROOT / "results" / run / "tb" / "*" / "events*")), key=lambda f: Path(f).stat().st_mtime)
    for file in files:
        acc = EventAccumulator(file, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag.startswith(prefix):
                series = out.setdefault(tag[len(prefix) :], {})
                for event in acc.Scalars(tag):
                    series[event.step] = event.value
    return out


def curve(series: dict[int, float]) -> tuple[list[int], list[float]]:
    steps = sorted(series)
    return steps, [series[s] for s in steps]


def figure_ablation() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True)
    data = {run: scalars(run, "eval/success/") for run, *_ in ARMS}

    for ax, task, title in zip(
        axes,
        ["mean", "peg-insert-side-v3", "pick-place-v3"],
        ["all 10 tasks (mean)", "peg-insert-side", "pick-place"],
    ):
        for run, label, colour, style in ARMS:
            if task not in data[run]:
                continue
            x, y = curve(data[run][task])
            ax.plot([s / 1e6 for s in x], y, style, color=colour, label=label, lw=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("success")
        ax.set_title(title, fontsize=8, pad=2)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("environment steps (millions)")
    axes[0].legend(fontsize=6, loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "mt10_ablation.pdf")
    plt.close(fig)


def figure_value_scale() -> None:
    """The mechanism: the per-task return scales the critic loss is divided by."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    scales = scalars("mt10_norm_curr_s0", "train/value_scale/")
    names = scalars("mt10_norm_curr_s0", "eval/success/")
    order = sorted(k for k in names if k != "mean")

    for tag, series in sorted(scales.items(), key=lambda kv: int(kv[0].removeprefix("task"))):
        index = int(tag.removeprefix("task"))
        label = order[index].replace("-v3", "") if index < len(order) else tag
        highlight = label in ("peg-insert-side", "pick-place")
        x, y = curve(series)
        ax.plot(
            [s / 1e6 for s in x],
            y,
            lw=1.4 if highlight else 0.7,
            color={"peg-insert-side": "tab:red", "pick-place": "tab:blue"}.get(label, "0.75"),
            label=label if highlight else None,
            zorder=3 if highlight else 1,
        )

    ax.set_yscale("log")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("value scale $\\sigma_i$")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "value_scale.pdf")
    plt.close(fig)


def report_numbers() -> None:
    """Print the figures' underlying numbers, so the text can quote them exactly."""
    for run, label, *_ in ARMS:
        data = scalars(run, "eval/success/")
        steps = sorted(data["mean"])
        print(f"\n== {run} ({label}): {steps[0]:,} -> {steps[-1]:,}")
        for at in (2_200_000, 3_400_000, 5_900_000):
            near = [s for s in steps if abs(s - at) < 60_000]
            if near:
                row = {t: data[t][near[0]] for t in data if near[0] in data[t]}
                hard = " ".join(
                    f"{t.replace('-v3', '')}={row[t]:.2f}"
                    for t in ("pick-place-v3", "peg-insert-side-v3")
                    if t in row
                )
                solved = sum(v >= 0.9 for t, v in row.items() if t != "mean")
                print(f"   ~{at / 1e6:.1f}M: mean={row['mean']:.3f} solved={solved} | {hard}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    figure_ablation()
    figure_value_scale()
    report_numbers()
    print(f"\nwrote {FIG}/mt10_ablation.pdf and {FIG}/value_scale.pdf")
