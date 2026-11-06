"""Train Agent B and evaluate it with exactly Agent A's protocol.

Run from the project root::

    python scripts\\train_agent_b.py
    python scripts\\train_agent_b.py --no-history --run-name agent_b_control

The rollout, opponent mix, evaluation opponents and interval are the same as
``train_agent_a.py``.  The only new inputs are the betting-history tokens, which
Agent B's GRU reads.  Keeping everything else fixed is what makes an A/B
comparison meaningful.

``--no-history`` trains the **control arm**: the same network, same initial
weights for a given seed, same heads, but the GRU output is replaced by zeros in
every forward pass.  It therefore receives no gradient.  Comparing the two arms on
identical deals answers "does the memory channel help?", while comparing Agent B
against Agent A answers "is the whole B agent better?".

``value`` is the mean squared value error divided by the rollout's return
variance, so ``0`` is a perfect critic and ``1`` predicts the mean.  ``EV`` is the
explained variance of the critic that collected the rollout.

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

from poker_arena.networks import PolicyValueNet
from poker_arena.networks_b import HISTORY_HIDDEN, HistoryPolicyValueNet
from poker_arena.ppo import PPOConfig
from poker_arena.training import OPPONENT_NAMES, EvaluationRecord
from poker_arena.training_b import TrainingConfigB, train_b


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(
            f"must be between 0 and 1, got {number}"
        )
    return number


def _opponent(value: str) -> str:
    if value not in OPPONENT_NAMES:
        raise argparse.ArgumentTypeError(
            f"unknown opponent {value!r}; choose from {', '.join(OPPONENT_NAMES)}"
        )
    return value


def _opponent_list(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one opponent name")
    unknown = [name for name in names if name not in OPPONENT_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown opponents {unknown}; choose from {', '.join(OPPONENT_NAMES)}"
        )
    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--iterations", type=_positive_int, default=50,
        help="collect-update iterations (default: 50)",
    )
    parser.add_argument(
        "--hands", type=_positive_int, default=320,
        help="hands per iteration, split by --self-play-fraction (default: 320)",
    )
    parser.add_argument(
        "--self-play-fraction", type=_fraction, default=0.5,
        help="fraction of each rollout played as self-play (default: 0.5)",
    )
    parser.add_argument(
        "--training-opponent", type=_opponent, default="raiser_strong",
        help="scripted opponent for the non-self-play hands "
             "(default: raiser_strong)",
    )
    parser.add_argument(
        "--eval-opponents", type=_opponent_list,
        default=("passive", "raiser_strong", "raiser_loose"),
        help="comma-separated opponents to evaluate "
             "(default: passive,raiser_strong,raiser_loose)",
    )
    parser.add_argument(
        "--eval-every", type=_positive_int, default=10,
        help="evaluate every N iterations (default: 10)",
    )
    parser.add_argument(
        "--eval-hands", type=_positive_int, default=2_000,
        help="hands per evaluation, per opponent (default: 2000)",
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
        "--run-name", default="agent_b",
        help="prefix for the log and checkpoint files (default: agent_b)",
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
    parser.add_argument(
        "--preflop-entropy-coef", type=float, default=0.0,
        help="extra entropy weight on pre-flop decisions only, keeping the "
             "opening action stochastic (default: 0.0)",
    )
    parser.add_argument(
        "--history-hidden", type=_positive_int, default=HISTORY_HIDDEN,
        help=f"width of the GRU state (default: {HISTORY_HIDDEN})",
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="control arm: zero the GRU output in every forward pass",
    )
    return parser.parse_args(argv)


def _config_from(args: argparse.Namespace) -> TrainingConfigB:
    return TrainingConfigB(
        iterations=args.iterations,
        hands_per_iteration=args.hands,
        self_play_fraction=args.self_play_fraction,
        training_opponent=args.training_opponent,
        evaluation_opponents=args.eval_opponents,
        evaluation_every=args.eval_every,
        evaluation_hands=args.eval_hands,
        seed=args.seed,
        evaluation_seed=args.eval_seed,
        run_name=args.run_name,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        ppo=PPOConfig(
            learning_rate=args.learning_rate,
            entropy_coef_preflop=args.preflop_entropy_coef,
        ),
        history_hidden=args.history_hidden,
        use_history=not args.no_history,
    )


def _format_row(row: dict[str, float | int], *, preflop: bool) -> str:
    preflop_column = ""
    if preflop:
        preflop_column = f"{float(row['entropy_preflop']):>9.4f}"
    return (
        f"{int(row['iteration']):>5}"
        f"{int(row['decisions']):>11}"
        f"{float(row['policy_loss']):>9.4f}"
        f"{float(row['value_loss']):>8.4f}"
        f"{float(row['explained_variance']):>8.2f}"
        f"{float(row['entropy']):>9.4f}"
        f"{preflop_column}"
        f"{float(row['approx_kl']):>8.4f}"
        f"{float(row['clip_fraction']):>7.2f}"
        f"{float(row['grad_norm']):>7.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from(args)

    print("Poker AI Arena - Agent B training")
    print(f"agent B parameters : {HistoryPolicyValueNet(history_hidden=config.history_hidden).count_parameters():,}"
          f"   agent A: {PolicyValueNet().count_parameters():,}")
    print(
        f"iterations {config.iterations}  hands/iter {config.hands_per_iteration}"
        f"  evaluate every {config.evaluation_every} on"
        f" {config.evaluation_hands} hands per opponent"
    )
    self_play_hands = int(round(
        config.hands_per_iteration * config.self_play_fraction
    ))
    print(
        f"mix      {self_play_hands} self-play"
        f" + {config.hands_per_iteration - self_play_hands}"
        f" vs {config.training_opponent!r}"
    )
    print(
        "eval     "
        + ", ".join(
            f"{name} ({'training' if name == config.training_opponent else 'held out'})"
            for name in config.evaluation_opponents
        )
    )
    print(
        f"history  {'on' if config.use_history else 'off (control arm)'}"
        f"  width {config.history_hidden}"
    )
    print(
        f"seed {config.seed}  eval seed {config.evaluation_seed}"
        f"  run name {config.run_name!r}"
    )
    print(
        f"ppo: clip {config.ppo.clip_epsilon}  entropy {config.ppo.entropy_coef}"
        f"  pre-flop entropy {config.ppo.entropy_coef_preflop}"
        f"  value {config.ppo.value_coef}  epochs {config.ppo.epochs}"
        f"  minibatch {config.ppo.minibatch_size}  lr {config.ppo.learning_rate}"
    )
    print()

    show_preflop = float(config.ppo.entropy_coef_preflop) > 0.0
    preflop_header = f"{'pre-ent':>9}" if show_preflop else ""
    print(
        f"{'iter':>5}{'decisions':>11}{'policy':>9}{'value':>8}{'EV':>8}"
        f"{'entropy':>9}{preflop_header}{'KL':>8}{'clip':>7}{'grad':>7}"
    )

    def progress(row, records) -> None:
        print(_format_row(row, preflop=show_preflop), flush=True)
        for record in records or ():
            result = record.result
            print(
                f"      eval @ {record.iteration:>4} vs {result.opponent:<14}"
                f" {result.mean_bb:+.4f} bb/hand "
                f"+/- {result.ci95_bb:.4f} (95%)  "
                f"total {result.total_bb:+.1f} bb over {result.hands} hands",
                flush=True,
            )

    result = train_b(config, progress=progress)

    print()
    print(f"train log      : {result.train_log}")
    print(f"evaluation log : {result.evaluation_log}")
    print(f"checkpoints    : {len(result.checkpoints)}")
    for path in result.checkpoints:
        print(f"                 {path}")

    if result.evaluations:
        print()
        print("evaluation summary (bb/hand, 95% interval)")
        by_opponent: dict[str, list[EvaluationRecord]] = {}
        for record in result.evaluations:
            by_opponent.setdefault(record.result.opponent, []).append(record)

        for name in config.evaluation_opponents:
            records = by_opponent.get(name, [])
            if not records:
                continue
            first = records[0]
            last = records[-1]
            intervals = [record.result.ci95_bb for record in records]
            print(
                f"  {name:<14} first @ {first.iteration:>4}: "
                f"{first.result.mean_bb:+.4f} +/- {first.result.ci95_bb:.4f}"
                f"   last @ {last.iteration:>4}: "
                f"{last.result.mean_bb:+.4f} +/- {last.result.ci95_bb:.4f}"
                f"   change {last.result.mean_bb - first.result.mean_bb:+.4f}"
                f"   typical +/- {sum(intervals) / len(intervals):.4f}"
            )
        print(
            "  a change smaller than its typical interval is not resolved "
            "by this sample"
        )
    else:
        print()
        print("no evaluation ran: raise --iterations above 0")

    assert len(result.iterations) == config.iterations
    assert all(Path(path).exists() for path in result.checkpoints)
    assert Path(result.train_log).exists() and Path(result.evaluation_log).exists()

    print("\nAgent B training run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
