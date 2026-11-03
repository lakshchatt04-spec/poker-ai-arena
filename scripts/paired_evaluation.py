"""Compare two checkpoints on exactly the same deals.

Run from the project root::

    python scripts\\paired_evaluation.py ^
        --checkpoint-a checkpoints\\agent_a_v3_0050.pt ^
        --checkpoint-b checkpoints\\agent_a_v3_0000.pt ^
        --opponent raiser_loose

Ordinary evaluation compares two means measured on *different* deals, so the
standard error of the difference is the square root of the two variances added.
This script replays each deal for both networks (``env.seed(seed + hand)``
before both), so the deal cancels out of the difference.  If the two policies
make similar decisions, most of the hand-to-hand payoff noise is common and the
paired interval is much tighter than the unpaired one.

The headline number is ``mean A - B``: positive means checkpoint A earned more
big blinds per hand than checkpoint B against the same opponent.  The script
prints both intervals so the size of the pairing gain is visible rather than
assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poker_arena.training import (  # noqa: E402
    OPPONENT_NAMES,
    evaluate_paired,
    load_checkpoint,
)


def _at_least_two(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {number}")
    return number


def _opponent(value: str) -> str:
    if value not in OPPONENT_NAMES:
        raise argparse.ArgumentTypeError(
            f"unknown opponent {value!r}; choose from {', '.join(OPPONENT_NAMES)}"
        )
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint-a", required=True,
        help="checkpoint whose payoff is being measured (A - B)",
    )
    parser.add_argument(
        "--checkpoint-b", required=True,
        help="checkpoint to compare against",
    )
    parser.add_argument(
        "--hands", type=_at_least_two, default=2_000,
        help="paired hands to play (default: 2000)",
    )
    parser.add_argument(
        "--opponent", type=_opponent, default="passive",
        help="scripted opponent both networks face (default: passive)",
    )
    parser.add_argument(
        "--seed", type=int, default=10_000,
        help="base deal seed; hand h uses seed + h (default: 10000)",
    )
    parser.add_argument(
        "--show-hands", type=_at_least_two, default=8,
        help="how many per-hand differences to print (default: 8)",
    )
    return parser.parse_args(argv)


def _load(path: str):

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    return load_checkpoint(checkpoint)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    net_a, meta_a = _load(args.checkpoint_a)
    net_b, meta_b = _load(args.checkpoint_b)

    result = evaluate_paired(
        net_a,
        net_b,
        args.hands,
        seed=args.seed,
        opponent_name=args.opponent,
    )

    print("Poker AI Arena - paired checkpoint comparison")
    print()
    print(f"checkpoint A : {args.checkpoint_a}  (iteration {meta_a['iteration']})")
    print(f"checkpoint B : {args.checkpoint_b}  (iteration {meta_b['iteration']})")
    print(f"opponent     : {args.opponent}")
    print(f"hands        : {result.hands}   base seed {args.seed}   "
          f"(hand h uses seed + h)")
    print()

    sample = min(args.show_hands, result.hands)
    print(f"per-hand difference A - B (positive = A won more), first {sample}:")
    print("  " + "  ".join(f"{value:+.1f}" for value in result.differences_bb[:sample]))
    print()

    print(f"mean A          : {result.mean_a_bb:+.4f} bb/hand")
    print(f"mean B          : {result.mean_b_bb:+.4f} bb/hand")
    print(f"mean A - B      : {result.mean_difference_bb:+.4f} bb/hand")
    print(f"paired 95%      : +/- {result.ci95_difference_bb:.4f} bb/hand "
          f"(stderr {result.stderr_difference_bb:.4f})")
    print(f"unpaired 95%    : +/- {1.959963984540054 * result.unpaired_stderr_bb:.4f} "
          f"bb/hand (stderr {result.unpaired_stderr_bb:.4f})")
    if result.stderr_difference_bb > 0.0:
        gain = result.unpaired_stderr_bb / result.stderr_difference_bb
        print(f"pairing gain    : {gain:.1f}x tighter than independent means")
    else:
        print("pairing gain    : infinite (the two networks tied on every deal)")
    print(f"wins/losses/ties: {result.wins} / {result.losses} / {result.ties}")
    print(f"total A - B     : {result.total_difference_bb:+.1f} bb over "
          f"{result.hands} hands")

    resolved = abs(result.mean_difference_bb) > result.ci95_difference_bb
    print()
    print("verdict         : "
          + ("resolved at 95% (the interval excludes zero)"
             if resolved else
             "not resolved at 95% (the interval includes zero)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
