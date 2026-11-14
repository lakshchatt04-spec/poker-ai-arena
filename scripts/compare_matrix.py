"""Run Agent B's whole comparison matrix and write one table.

``compare_agents.py`` answers one question about one pair of checkpoints.  The
experiment needs the same three questions answered for every seed and every
evaluation opponent, so this script runs them in a single process (loading the
networks once instead of twenty-seven times) and aggregates across seeds.

The three questions, all paired on identical deals:

``history-vs-control``
    ``agent_b_seed{s}`` minus ``agent_b_control_seed{s}``.  The control arm has
    the same trunk, the same heads and the same initial weights; only the GRU
    input is missing.  This is the clean test of whether memory helps.

``agent-a-vs-b``
    ``agent_a_seed{s}`` minus ``agent_b_seed{s}``.  Agent A is frozen, so this is
    the headline architecture comparison.  B is also 1.16x larger, so a B win is
    not automatically a memory win.

``ablation``
    ``agent_b_seed{s}`` with its history channel on minus the same weights with
    it zeroed.  Identical weights and identical deals: it measures how much the
    trained policy *relies* on the memory channel, not whether memory helped
    during training.

Positive always means the first policy out-earned the second.

Usage::

    python scripts\\compare_matrix.py --hands 2000
    python scripts\\compare_matrix.py --hands 2000 --out runs\\b_matrix.md
    python scripts\\compare_matrix.py --iteration 100 --modes history-vs-control ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

from poker_arena.encoding import ObservationEncoder
from poker_arena.environment import make_environment
from poker_arena.training import OPPONENT_NAMES
from poker_arena.training_b import (
    AgentBPolicy,
    evaluate_paired_policies,
)
from compare_agents import load_agent, make_policy

MODES = ("history-vs-control", "agent-a-vs-b", "ablation")

#: Two-sided 95 % t quantile for n - 1 = 2 degrees of freedom.
T95_DF2 = 4.302652729749464


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hands", type=int, default=2_000,
        help="paired hands per comparison (default: 2000)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="training seeds to compare (default: 0 1 2)",
    )
    parser.add_argument(
        "--opponents", nargs="+", default=list(OPPONENT_NAMES),
        choices=list(OPPONENT_NAMES),
        help="scripted opponents (default: all three)",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(MODES), choices=list(MODES),
        help="comparisons to run (default: all three)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints"),
        help="where the .pt files live",
    )
    parser.add_argument(
        "--history-prefix", default="agent_b",
        help="run-name prefix of the history arm (default: agent_b)",
    )
    parser.add_argument(
        "--control-prefix", default="agent_b_control",
        help="run-name prefix of the control arm (default: agent_b_control)",
    )
    parser.add_argument(
        "--iteration", type=int, default=50,
        help="checkpoint iteration to compare (default: 50)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=10_000,
        help="base deal seed; hand h uses seed + h (default: 10000)",
    )
    parser.add_argument(
        "--out", default=None,
        help="write the markdown table here as well as printing it",
    )
    parser.add_argument(
        "--save-differences", action="store_true",
        help="also write every per-hand difference to runs/b_matrix_differences.csv",
    )
    return parser.parse_args(argv)


def _checkpoint_paths(
    mode: str,
    seed: int,
    directory: Path,
    *,
    history_prefix: str = "agent_b",
    control_prefix: str = "agent_b_control",
    iteration: int = 50,
) -> tuple[Path, Path]:
    suffix = f"_{iteration:04d}.pt"
    history = directory / f"{history_prefix}_seed{seed}{suffix}"
    if mode == "history-vs-control":
        return history, directory / f"{control_prefix}_seed{seed}{suffix}"
    if mode == "agent-a-vs-b":
        return directory / f"agent_a_seed{seed}{suffix}", history
    if mode == "ablation":
        return history, history
    raise ValueError(f"unknown mode {mode!r}")


def _policies(
    mode: str, first: Path, second: Path, encoder: ObservationEncoder
):
    """Build the two policies for one comparison, or ``None`` if files are absent."""

    if not first.is_file() or not second.is_file():
        return None

    net_first, meta_first, kind_first = load_agent(first)
    policy_first = make_policy(net_first, kind_first, encoder)

    if mode == "ablation":
        policy_second = AgentBPolicy(net_first, encoder, use_history=False)
        return policy_first, policy_second, meta_first

    net_second, meta_second, kind_second = load_agent(second)
    policy_second = make_policy(net_second, kind_second, encoder)
    return policy_first, policy_second, meta_first


def _aggregate(values: list[float]) -> tuple[float, float, float]:
    """Return ``(mean, sd, 95% half-width)`` across seeds."""

    data = np.asarray(values, dtype=np.float64)
    if data.size == 1:
        return float(data.mean()), float("nan"), float("nan")
    sd = float(data.std(ddof=1))
    half = T95_DF2 * sd / np.sqrt(data.size)
    return float(data.mean()), sd, float(half)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.hands < 2:
        raise SystemExit("--hands must be at least 2")

    directory = Path(args.checkpoint_dir)
    env = make_environment(seed=args.seed_base)
    encoder = ObservationEncoder.from_environment(env)

    if args.save_differences:
        stale = PROJECT_ROOT / "runs" / "b_matrix_differences.csv"
        if stale.is_file():
            stale.unlink()

    lines: list[str] = []
    csv_rows: list[str] = ["mode,opponent,seed,mean_difference_bb,ci95_bb,hands"]

    for mode in args.modes:
        lines.append(f"## {mode} @ iteration {args.iteration}")
        lines.append("")
        lines.append("| opponent | seed | first - second | 95 % | resolved |")
        lines.append("|---|---|---|---|---|")

        for opponent in args.opponents:
            seed_means: list[float] = []
            for seed in args.seeds:
                first, second = _checkpoint_paths(
                    mode, seed, directory,
                    history_prefix=args.history_prefix,
                    control_prefix=args.control_prefix,
                    iteration=args.iteration,
                )
                built = _policies(mode, first, second, encoder)
                if built is None:
                    lines.append(
                        f"| {opponent} | {seed} | *missing checkpoint* | n/a | n/a |"
                    )
                    continue

                policy_first, policy_second, _ = built
                result = evaluate_paired_policies(
                    policy_first,
                    policy_second,
                    args.hands,
                    seed=args.seed_base,
                    opponent_name=opponent,
                )
                resolved = abs(result.mean_difference_bb) > result.ci95_difference_bb
                seed_means.append(result.mean_difference_bb)
                lines.append(
                    f"| {opponent} | {seed} | "
                    f"{result.mean_difference_bb:+.4f} | "
                    f"+/-{result.ci95_difference_bb:.4f} | "
                    f"{'yes' if resolved else 'no'} |"
                )
                csv_rows.append(
                    f"{mode},{opponent},{seed},"
                    f"{result.mean_difference_bb:.6f},"
                    f"{result.ci95_difference_bb:.6f},{result.hands}"
                )
                if args.save_differences:
                    differences_path = PROJECT_ROOT / "runs" / "b_matrix_differences.csv"
                    differences_path.parent.mkdir(parents=True, exist_ok=True)
                    with differences_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"# {mode},{opponent},seed={seed}\n"
                        )
                        handle.write(
                            "\n".join(f"{value:.6f}" for value in result.differences_bb)
                        )
                        handle.write("\n")

            if len(seed_means) == len(args.seeds) and len(seed_means) > 1:
                mean, sd, half = _aggregate(seed_means)
                agree = sum(
                    1 for value in seed_means if (value > 0.0) == (mean > 0.0)
                )
                lines.append(
                    f"| {opponent} | **mean** | **{mean:+.4f}** | "
                    f"sd {sd:.4f}, +/-{half:.4f} | "
                    f"{agree}/{len(seed_means)} seeds same sign |"
                )
        lines.append("")

    table = "\n".join(lines)
    print(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "Paired comparison matrix (bb/hand; positive = first minus second)\n\n"
            + table,
            encoding="utf-8",
        )
        print(f"table written to {out_path}")

    if args.save_differences:
        csv_path = PROJECT_ROOT / "runs" / "b_matrix.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
        print(f"summary csv written to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
