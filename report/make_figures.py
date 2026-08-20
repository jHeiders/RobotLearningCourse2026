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

# The comparison the report makes: uniform replay against the failure-rate curriculum.
ARMS = [
    ("mt10_s0", "baseline", "0.55", "-"),
    ("mt10_curr_s0", "curriculum", "tab:orange", "-"),
]
HARD = [("peg-insert-side-v3", "peg-insert-side"), ("pick-place-v3", "pick-place")]
BUDGET = 9.0  # Mstep; the common budget every arm is compared at in the tables.
XMAX = 9.0
XLABEL = "environment steps (millions)"  # identical on every figure, per review
HARD_COLOUR = {"peg-insert-side": "tab:red", "pick-place": "tab:blue"}

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


def curve(series: dict[int, float], limit: float = XMAX) -> tuple[list[float], list[float]]:
    steps = [s for s in sorted(series) if s / 1e6 <= limit]
    return [s / 1e6 for s in steps], [series[s] for s in steps]


def _finish(ax, ylabel: str, xlabel: bool = True) -> None:
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(0, XMAX)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(XLABEL)
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
    """The two slow tasks, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1), sharey=True)
    for ax, (task, title) in zip(axes, HARD):
        for run, label, colour, style in ARMS:
            if run not in data or task not in data[run]:
                continue
            ax.plot(*curve(data[run][task]), style, color=colour, label=label)
        ax.set_title(title)
        _finish(ax, "", xlabel=False)
    axes[0].set_ylabel("success")
    # Both panels have curves in every corner, so the legend goes under the axes
    # rather than on top of the data. One shared x label, worded as on every other figure.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0.18, 1, 1))
    fig.supxlabel(XLABEL, fontsize=8, y=0.15)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.savefig(FIG / "mt10_hard.pdf")
    plt.close(fig)


def figure_share(_data) -> None:
    """The mechanism: what fraction of each batch the curriculum gives each task."""
    shares = scalars("mt10_curr_s0", "curriculum/share/")
    if not shares:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for tag, series in shares.items():
        name = tag.replace("-v3", "")
        colour = HARD_COLOUR.get(name)
        ax.plot(*curve(series), color=colour or "0.75", lw=1.4 if colour else 0.7,
                label=name if colour else None, zorder=3 if colour else 1)
    ax.axhline(0.1, color="0.35", ls=":", lw=0.8, zorder=2)
    # Below the line and to the right: the only region no curve passes through.
    ax.annotate("uniform share (1/10)", xy=(XMAX * 0.42, 0.028), fontsize=6, color="0.35")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("share of each batch")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "curriculum_share.pdf")
    plt.close(fig)


def _row(data, run: str, limit: float) -> str:
    steps = [s for s in data[run]["mean"] if s / 1e6 <= limit]
    if not steps:
        return f"{'-- nothing at this budget --':>36}"
    last = max(steps)
    row = {t: v[last] for t, v in data[run].items() if last in v}
    solved = sum(v >= 0.9 for t, v in row.items() if t != "mean")
    return (
        f"{row['mean']:>7.3f}{solved:>8}"
        f"{row.get('peg-insert-side-v3', float('nan')):>7.2f}"
        f"{row.get('pick-place-v3', float('nan')):>7.2f}   @{last:,}"
    )


def report_numbers(data) -> None:
    """Print each arm's numbers, so the tables can quote them."""
    print(f"\nat each arm's endpoint\n{'arm':<24}{'mean':>7}{'solved':>8}{'peg':>7}{'pick':>7}")
    for run, label, *_ in ARMS:
        print(f"{label:<24}" + (_row(data, run, BUDGET) if run in data else "-- not run --"))

    # The best single evaluation of each arm, which is what the text quotes alongside the
    # endpoint: the two curriculum arms both peak well above where they happen to end.
    print("\nbest single evaluation")
    for run, label, *_ in ARMS:
        if run not in data or not data[run]:
            continue
        mean = data[run]["mean"]
        step = max(mean, key=lambda s: mean[s])
        solved = sum(v[step] >= 0.9 for t, v in data[run].items() if t != "mean" and step in v)
        print(f"{label:<24}{mean[step]:>7.3f}{solved:>8}   @{step:,}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    data = {run: scalars(run, "eval/success/") for run, *_ in ARMS}
    data = {run: series for run, series in data.items() if series}
    figure_mean(data)
    figure_hard(data)
    figure_share(data)
    report_numbers(data)
    print(f"\nwrote {FIG}/mt10_mean.pdf, mt10_hard.pdf, curriculum_share.pdf")
