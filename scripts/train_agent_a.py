"""Train Agent A and evaluate it against a fixed scripted opponent.

Run from the project root::

    python scripts\\train_agent_a.py

Every ``--eval-every`` iterations (and on the final iteration) the current
network plays ``--eval-hands`` hands against a scripted call/check opponent,
using the same deals and the same seed every time.  A per-hand payoff is noisy,
so the evaluation column is a mean with a 95 % interval, never a bare number.

Results are written to ``runs/<run-name>_train.csv``,
``runs/<run-name>_eval.csv`` and ``checkpoints/<run-name>_<iter>.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poker_arena.ppo import PPOConfig
from poker_arena.training import TrainingConfig, train


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--iterations", type=_positive_int, default=50,
        help="collect-update iterations (default: 50)",
    )
    parser.add_argument(
        "--hands", type=_positive_int, default=320,
        help="self-play hands per iteration (default: 320)",
    )
    parser.add_argument(
        "--eval-every", type=_positive_int, default=10,
        help="evaluate every N iterations (default: 10)",
    )
    parser.add_argument(
        "--eval-hands", type=_positive_int, default=2_000,
        help="hands per evaluation (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="training seed (default: 0)",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=10_000,
        help="evaluation seed, fixed across runs (default: 10000)",
    )
    parser.add_argument(
        "--run-name", default="agent_a",
        help="prefix for the log and checkpoint files (default: agent_a)",
    )
    parser.add_argument(
        "--log-dir", default="runs",
        help="directory for the CSV logs (default: runs)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="directory for the checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Adam learning rate (default: 3e-4)",
    )
    return parser.parse_args(argv)


def _config_from(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        iterations=args.iterations,
        hands_per_iteration=args.hands,
        evaluation_every=args.eval_every,
        evaluation_hands=args.eval_hands,
        seed=args.seed,
        evaluation_seed=args.eval_seed,
        run_name=args.run_name,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        ppo=PPOConfig(learning_rate=args.learning_rate),
    )


def _format_row(row: dict[str, float | int]) -> str:
    return (
        f"{int(row['iteration']):>5}"
        f"{int(row['decisions']):>11}"
        f"{float(row['policy_loss']):>9.4f}"
        f"{float(row['value_loss']):>10.4f}"
        f"{float(row['entropy']):>9.4f}"
        f"{float(row['approx_kl']):>8.4f}"
        f"{float(row['clip_fraction']):>7.2f}"
        f"{float(row['grad_norm']):>7.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from(args)

    print("Poker AI Arena - Agent A training")
    print(
        f"iterations {config.iterations}  hands/iter {config.hands_per_iteration}"
        f"  evaluate every {config.evaluation_every} on"
        f" {config.evaluation_hands} hands"
    )
    print(
        f"seed {config.seed}  eval seed {config.evaluation_seed}"
        f"  run name {config.run_name!r}"
    )
    print(
        f"ppo: clip {config.ppo.clip_epsilon}  entropy {config.ppo.entropy_coef}"
        f"  value {config.ppo.value_coef}  epochs {config.ppo.epochs}"
        f"  minibatch {config.ppo.minibatch_size}  lr {config.ppo.learning_rate}"
    )
    print()

    print(
        f"{'iter':>5}{'decisions':>11}{'policy':>9}{'value':>10}"
        f"{'entropy':>9}{'KL':>8}{'clip':>7}{'grad':>7}"
    )

    def progress(row, record) -> None:
        print(_format_row(row), flush=True)
        if record is not None:
            result = record.result
            print(
                f"      eval @ {record.iteration:>4}: "
                f"{result.mean_bb:+.4f} bb/hand "
                f"+/- {result.ci95_bb:.4f} (95%)  "
                f"total {result.total_bb:+.1f} bb over {result.hands} hands",
                flush=True,
            )

    result = train(config, progress=progress)

    print()
    print(f"train log      : {result.train_log}")
    print(f"evaluation log : {result.evaluation_log}")
    print(f"checkpoints    : {len(result.checkpoints)}")
    for path in result.checkpoints:
        print(f"                 {path}")

    if result.evaluations:
        first = result.evaluations[0].result
        last = result.evaluations[-1].result
        print()
        print("evaluation summary (bb/hand, 95% interval)")
        print(
            f"  first @ {result.evaluations[0].iteration:>4}: "
            f"{first.mean_bb:+.4f} +/- {first.ci95_bb:.4f}"
        )
        print(
            f"  last  @ {result.evaluations[-1].iteration:>4}: "
            f"{last.mean_bb:+.4f} +/- {last.ci95_bb:.4f}"
        )
        print(
            f"  change          : {last.mean_bb - first.mean_bb:+.4f} bb/hand"
        )
        intervals = [record.result.ci95_bb for record in result.evaluations]
        print(
            f"  typical interval: +/- {sum(intervals) / len(intervals):.4f}; "
            "a change smaller than this is not resolved by the sample"
        )
    else:
        print()
        print("no evaluation ran: raise --iterations above 0")

    assert len(result.iterations) == config.iterations
    assert all(
        Path(path).exists() for path in result.checkpoints
    ), "a checkpoint was reported but not written"
    assert Path(result.train_log).exists() and Path(result.evaluation_log).exists()

    print("\nAgent A training run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
