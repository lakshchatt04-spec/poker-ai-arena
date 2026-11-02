"""Plot a training run's CSV logs.

Run from the project root after ``train_agent_a.py``::

    python scripts\\plot_training.py

Reads ``runs/<run-name>_train.csv`` and ``runs/<run-name>_eval.csv`` and writes
``runs/<run-name>_curve.png``.

The layout is deliberate: the evaluation panel answers "is it getting better?",
the value/entropy panel answers "is it still learning?", and the two diagnostics
panels answer "is the update still under control?". A single loss curve would
mix all three questions into one unreadable line.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")  # write files; do not require a display

import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLUMNS = (
    "iteration",
    "decisions",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nrun scripts\\train_agent_a.py first, or pass "
            f"--log-dir / --run-name"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="agent_a")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument(
        "--output", default=None,
        help="PNG path (default: <log-dir>/<run-name>_curve.png)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log_dir = Path(args.log_dir)
    train_path = log_dir / f"{args.run_name}_train.csv"
    eval_path = log_dir / f"{args.run_name}_eval.csv"
    output = (
        Path(args.output)
        if args.output
        else log_dir / f"{args.run_name}_curve.png"
    )

    train_rows = _read_csv(train_path)
    if not train_rows:
        raise SystemExit(f"{train_path} has no rows")
    missing = [name for name in TRAIN_COLUMNS if name not in train_rows[0]]
    if missing:
        raise SystemExit(f"{train_path} is missing columns: {missing}")

    eval_rows = _read_csv(eval_path) if eval_path.exists() else []

    iteration = _column(train_rows, "iteration")
    decisions = _column(train_rows, "decisions")
    value_loss = _column(train_rows, "value_loss")
    entropy = _column(train_rows, "entropy")
    approx_kl = _column(train_rows, "approx_kl")
    clip_fraction = _column(train_rows, "clip_fraction")
    grad_norm = _column(train_rows, "grad_norm")

    eval_iteration = _column(eval_rows, "iteration") if eval_rows else np.array([])
    eval_mean = _column(eval_rows, "mean_bb") if eval_rows else np.array([])
    eval_ci = _column(eval_rows, "ci95_bb") if eval_rows else np.array([])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    figure.suptitle(
        f"Agent A - run {args.run_name!r} ({len(train_rows)} iterations)",
        fontsize=13,
    )

    def mark_evaluations(axis) -> None:
        for x in eval_iteration:
            axis.axvline(x, color="0.75", linewidth=0.8, zorder=0)

    # -- evaluation: the only panel that should be read as "progress" ------
    top_left = axes[0, 0]
    mark_evaluations(top_left)
    if eval_rows:
        top_left.errorbar(
            eval_iteration,
            eval_mean,
            yerr=eval_ci,
            fmt="o-",
            color="tab:blue",
            capsize=4,
            linewidth=1.5,
            markersize=5,
        )
        top_left.annotate(
            f"{eval_mean[-1]:+.3f}",
            xy=(eval_iteration[-1], eval_mean[-1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="tab:blue",
        )
        best = int(np.argmax(eval_mean))
        top_left.annotate(
            f"best {eval_mean[best]:+.3f}",
            xy=(eval_iteration[best], eval_mean[best]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=9,
            color="tab:green",
        )
    else:
        top_left.text(
            0.5, 0.5, "no evaluations in this run",
            ha="center", va="center", transform=top_left.transAxes,
            color="0.4",
        )
    top_left.axhline(0.0, color="black", linewidth=0.9)
    top_left.set_title("Evaluation vs scripted opponent (95% interval)")
    top_left.set_ylabel("bb per hand")

    # -- the two learning signals ------------------------------------------
    top_right = axes[0, 1]
    mark_evaluations(top_right)
    top_right.plot(iteration, value_loss, color="tab:red", linewidth=1.4)
    top_right.set_title("Value loss and policy entropy")
    top_right.set_ylabel("value loss", color="tab:red")
    top_right.tick_params(axis="y", labelcolor="tab:red")

    entropy_axis = top_right.twinx()
    entropy_axis.plot(
        iteration, entropy, color="tab:purple", linewidth=1.4, linestyle="--"
    )
    entropy_axis.set_ylabel("entropy (nats)", color="tab:purple")
    entropy_axis.tick_params(axis="y", labelcolor="tab:purple")
    entropy_axis.set_ylim(bottom=0.0)

    # -- gradient norm against its limit -----------------------------------
    bottom_left = axes[1, 0]
    mark_evaluations(bottom_left)
    bottom_left.plot(iteration, grad_norm, color="tab:orange", linewidth=1.2)
    bottom_left.axhline(
        0.5, color="black", linewidth=0.9, linestyle="--",
        label="limit 0.5",
    )
    bottom_left.set_title("Gradient norm before clipping")
    bottom_left.set_ylabel("L2 norm")
    bottom_left.set_xlabel("iteration")
    bottom_left.legend(loc="upper right", frameon=False, fontsize=9)

    # -- trust region ------------------------------------------------------
    bottom_right = axes[1, 1]
    mark_evaluations(bottom_right)
    bottom_right.plot(iteration, approx_kl, color="tab:cyan", linewidth=1.4)
    bottom_right.set_title("Trust region: approx KL and clip fraction")
    bottom_right.set_ylabel("approx KL", color="tab:cyan")
    bottom_right.tick_params(axis="y", labelcolor="tab:cyan")
    bottom_right.set_xlabel("iteration")
    if float(np.max(approx_kl)) > 0.0:
        bottom_right.set_ylim(bottom=0.0)

    clip_axis = bottom_right.twinx()
    clip_axis.plot(
        iteration, clip_fraction, color="tab:brown", linewidth=1.2,
        linestyle=":",
    )
    clip_axis.set_ylabel("clip fraction", color="tab:brown")
    clip_axis.tick_params(axis="y", labelcolor="tab:brown")
    clip_axis.set_ylim(-0.02, 1.02)

    for axis in axes.flat:
        axis.grid(alpha=0.2)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)

    # -- text summary ------------------------------------------------------
    print(f"run            : {args.run_name}")
    print(f"iterations     : {len(train_rows)}")
    print(f"decisions      : {int(decisions.sum())} total, "
          f"{decisions.mean():.0f} per iteration")
    print(f"value loss     : {value_loss[0]:.4f} -> {value_loss[-1]:.4f}")
    print(f"entropy        : {entropy[0]:.4f} -> {entropy[-1]:.4f}")
    print(f"approx KL      : max {approx_kl.max():.5f}, "
          f"mean {approx_kl.mean():.5f}")
    print(f"clip fraction  : max {clip_fraction.max():.3f}")
    print(f"grad norm      : max {grad_norm.max():.3f}, "
          f"over limit on {int((grad_norm > 0.5).sum())} iterations")
    if eval_rows:
        print()
        print(f"{'iter':>6}{'hands':>8}{'mean bb':>10}{'95%':>9}")
        for record, mean, ci in zip(eval_rows, eval_mean, eval_ci):
            print(
                f"{int(float(record['iteration'])):>6}"
                f"{int(float(record['hands'])):>8}"
                f"{mean:>+10.4f}{ci:>9.4f}"
            )
        print()
        print(f"first eval     : {eval_mean[0]:+.4f} bb/hand")
        print(f"last eval      : {eval_mean[-1]:+.4f} bb/hand")
        print(f"change         : {eval_mean[-1] - eval_mean[0]:+.4f} bb/hand")
        print(f"best eval      : {eval_mean.max():+.4f} "
              f"at iteration {int(eval_iteration[int(np.argmax(eval_mean))])}")
        print(f"typical 95%    : +/- {eval_ci.mean():.4f} bb/hand")

    print(f"\nplot written to {output.resolve()}")
    print("Training plot complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
