"""Regenerate the report's figures from the TensorBoard logs in results/.

    python report/make_figures.py

Arms that have not been run yet are skipped, so this can be re-run as the ablation fills
in. Every event file of a run is read in modification order, so a resumed run's re-logged
steps replace the originals rather than appearing twice.
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

# The 2x2: replay curriculum on/off x normalised critic loss on/off.
ARMS = [
    ("mt10_s0", "neither", "0.55", "-"),
    ("mt10_curr_scratch_s0", "curriculum", "tab:orange", "-"),
    ("mt10_norm_s0", "normalisation", "tab:green", "-"),
    ("mt10_norm_curr_s0", "both", "tab:blue", "-"),
]
HARD = [("peg-insert-side-v3", "peg-insert-side"), ("pick-place-v3", "pick-place")]
BUDGET = 6.0  # Mstep; arms are compared at a common budget even if one ran longer.

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
    }
)


def scalars(run: str, prefix: str) -> dict[str, dict[int, float]]:
    """tag suffix -> {step: value}, later event files winning on duplicate steps."""
    out: dict[str, dict[int, float]] = {}
    files = glob.glob(str(ROOT / "results" / run / "tb" / "*" / "events*"))
    for file in sorted(files, key=lambda f: Path(f).stat().st_mtime):
        acc = EventAccumulator(file, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag.startswith(prefix):
                series = out.setdefault(tag[len(prefix) :], {})
                for event in acc.Scalars(tag):
                    series[event.step] = event.value
    return out


def curve(series: dict[int, float], limit: float = BUDGET) -> tuple[list[float], list[float]]:
    steps = [s for s in sorted(series) if s / 1e6 <= limit]
    return [s / 1e6 for s in steps], [series[s] for s in steps]


def _finish(ax, ylabel: str, xlabel: bool = True) -> None:
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(0, BUDGET)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("environment steps (millions)")
    ax.grid(axis="y", alpha=0.25, lw=0.4)


def figure_mean(data) -> None:
    """Headline: what the four arms reach overall."""
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for run, label, colour, style in ARMS:
        if run not in data:
            continue
        ax.plot(*curve(data[run]["mean"]), style, color=colour, label=label)
    _finish(ax, "mean success (10 tasks)")
    ax.legend(loc="lower right", frameon=False, ncol=2, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "mt10_mean.pdf")
    plt.close(fig)


def figure_hard(data) -> None:
    """The two tasks the baseline never learns, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    for ax, (task, title) in zip(axes, HARD):
        for run, label, colour, style in ARMS:
            if run not in data or task not in data[run]:
                continue
            ax.plot(*curve(data[run][task]), style, color=colour, label=label)
        ax.set_title(title)
        _finish(ax, "")
        ax.set_xlabel("Msteps")
    axes[0].set_ylabel("success")
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(FIG / "mt10_hard.pdf")
    plt.close(fig)


def figure_value_scale() -> None:
    """The mechanism: the per-task return scales the critic loss is divided by."""
    scales = scalars("mt10_norm_curr_s0", "train/value_scale/")
    names = [k for k in scalars("mt10_norm_curr_s0", "eval/success/") if k != "mean"]
    if not scales:
        return
    order = sorted(names)

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for tag, series in scales.items():
        index = int(tag.removeprefix("task"))
        label = order[index].replace("-v3", "") if index < len(order) else tag
        hard = label in ("peg-insert-side", "pick-place")
        colour = {"peg-insert-side": "tab:red", "pick-place": "tab:blue"}.get(label, "0.8")
        x, y = curve(series)
        ax.plot(x, y, lw=1.5 if hard else 0.7, color=colour,
                label=label if hard else None, zorder=3 if hard else 1)

    ax.set_yscale("log")
    ax.set_xlim(0, BUDGET)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel(r"value scale $\sigma_i$")
    ax.grid(axis="y", alpha=0.25, lw=0.4, which="major")
    ax.legend(loc="lower right", frameon=False)
    ax.annotate(
        "the eight tasks the baseline solves",
        xy=(BUDGET * 0.55, 45), fontsize=6, color="0.45", ha="center",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "value_scale.pdf")
    plt.close(fig)


def report_numbers(data) -> None:
    """Print each arm's numbers at the common budget, so the tables can quote them."""
    print(f"\n{'arm':<24}{'mean':>7}{'solved':>8}{'peg':>7}{'pick':>7}   (at <= 6M)")
    for run, label, *_ in ARMS:
        if run not in data:
            print(f"{label:<24}{'-- not run yet --':>29}")
            continue
        steps = [s for s in data[run]["mean"] if s / 1e6 <= BUDGET]
        last = max(steps)
        row = {t: v[last] for t, v in data[run].items() if last in v}
        solved = sum(v >= 0.9 for t, v in row.items() if t != "mean")
        print(
            f"{label:<24}{row['mean']:>7.3f}{solved:>8}"
            f"{row.get('peg-insert-side-v3', float('nan')):>7.2f}"
            f"{row.get('pick-place-v3', float('nan')):>7.2f}   @{last:,}"
        )


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    data = {run: scalars(run, "eval/success/") for run, *_ in ARMS}
    data = {run: series for run, series in data.items() if series}
    figure_mean(data)
    figure_hard(data)
    figure_value_scale()
    report_numbers(data)
    print(f"\nwrote {FIG}/mt10_mean.pdf, mt10_hard.pdf, value_scale.pdf")
