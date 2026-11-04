"""Compare two checkpoints -- Agent A, Agent B, or the B history ablation.

Three uses, all on the *same deals*:

1. Agent A against Agent B::

       python scripts\\compare_agents.py ^
           --a checkpoints\\agent_a_seed0_0050.pt ^
           --b checkpoints\\agent_b_seed0_0050.pt ^
           --opponent raiser_loose

2. Agent B's history channel against the same weights with it zeroed::

       python scripts\\compare_agents.py ^
           --b checkpoints\\agent_b_seed0_0050.pt ^
           --ablate --opponent raiser_loose

   This is the within-network control: identical weights, identical deals, only
   the GRU input changes.  It measures how much the trained policy relies on
   memory, though the heads were trained with it, so it is a reliance test rather
   than a clean counterfactual.

3. Agent B trained with history against Agent B trained without it (the control
   arm).  Because the network is constructed identically for a given seed, the two
   runs start from the same weights; only the input differs::

       python scripts\\compare_agents.py ^
           --a checkpoints\\agent_b_seed0_0050.pt ^
           --b checkpoints\\agent_b_control_seed0_0050.pt

The checkpoint type is detected from the payload: Agent B's checkpoints record
``history_hidden``, Agent A's do not.

Each hand is replayed after ``env.seed(seed + hand)`` for both policies, and a
stochastic scripted opponent is rebuilt per play from the same per-hand seed, so
the cards, blinds and opponent draws are shared.  The headline is ``mean A - B``;
positive means A earned more big blinds per hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from poker_arena.encoding import ObservationEncoder
from poker_arena.environment import make_environment
from poker_arena.training import (
    OPPONENT_NAMES,
    load_checkpoint,
)
from poker_arena.training_b import (
    AgentAPolicy,
    AgentBPolicy,
    evaluate_paired_policies,
    load_checkpoint_b,
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
        "--a", default=None,
        help="first checkpoint (Agent A or Agent B); omit to use --ablate",
    )
    parser.add_argument(
        "--b", required=True,
        help="second checkpoint (Agent B)",
    )
    parser.add_argument(
        "--ablate", action="store_true",
        help="ignore --a; compare --b with history against the same weights "
             "with history zeroed",
    )
    parser.add_argument(
        "--hands", type=_at_least_two, default=2_000,
        help="paired hands to play (default: 2000)",
    )
    parser.add_argument(
        "--opponent", type=_opponent, default="passive",
        help="scripted opponent both policies face (default: passive)",
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


def load_agent(path: str):
    """Load either checkpoint type and return ``(net, metadata, kind)``."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "history_hidden" in payload:
        net, metadata = load_checkpoint_b(checkpoint)
        return net, metadata, "agent_b"

    net, metadata = load_checkpoint(checkpoint)
    return net, metadata, "agent_a"


def make_policy(net, kind: str, encoder, *, use_history: bool = True):
    """Wrap a loaded network in the policy that plays it greedily."""
    if kind == "agent_b":
        return AgentBPolicy(net, encoder, use_history=use_history)
    return AgentAPolicy(net, encoder)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.ablate:
        if args.a is not None:
            raise SystemExit("--ablate compares --b with itself; do not pass --a")
        a_path = b_path = args.b
    else:
        if args.a is None:
            raise SystemExit("pass --a for a two-checkpoint comparison, or --ablate")
        a_path = args.a
        b_path = args.b

    env = make_environment(seed=args.seed)
    encoder = ObservationEncoder.from_environment(env)

    net_a, meta_a, kind_a = load_agent(a_path)
    net_b, meta_b, kind_b = load_agent(b_path)

    policy_a = make_policy(net_a, kind_a, encoder)
    policy_b = make_policy(net_b, kind_b, encoder)
    label_a = a_path
    label_b = b_path
    if args.ablate:
        policy_a = AgentBPolicy(net_b, encoder, use_history=True)
        policy_b = AgentBPolicy(net_b, encoder, use_history=False)
        label_a = f"{b_path}  (history on)"
        label_b = f"{b_path}  (history off)"

    result = evaluate_paired_policies(
        policy_a,
        policy_b,
        args.hands,
        seed=args.seed,
        opponent_name=args.opponent,
    )

    print("Poker AI Arena - paired agent comparison")
    print()
    print(f"A : {label_a}  ({kind_a}, iteration {meta_a['iteration']})")
    print(f"B : {label_b}  ({kind_b}, iteration {meta_b['iteration']})")
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
        print("pairing gain    : infinite (the two policies tied on every deal)")
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
