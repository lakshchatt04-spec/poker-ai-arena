"""How much does Agent B's policy actually depend on the history channel?

The paired EV comparison is the headline, but per-hand payoff differences are
noisy: at 2,000 hands the interval is about +/-0.15 bb/hand, which is the same
size as any effect we could realistically expect.  This script measures reliance
directly and with a much smaller variance.

The protocol avoids the trap that makes a naive "run both and diff" test
uninformative (once the two policies act differently, they are no longer in the
same states):

1. Play ``--hands`` deals with the history channel **on** and record every
   decision state (features, token table, mask, legal mask) as it is reached.
2. Replay those *same* recorded states through the *same* weights twice, once
   with ``use_history=True`` and once with ``use_history=False``.
3. Count how often the greedy action changes.

That is a pure counterfactual on a fixed state distribution.  A policy that
learned to use memory will change its action on a non-trivial fraction of
decisions; a policy that ignores the channel will change on almost none.

The zero-history row is reported as a baseline: it is the same test on the
control arm's weights, and it should be near zero because those weights never
saw the history input.

Usage::

    python scripts\\history_reliance.py
    python scripts\\history_reliance.py --hands 1000 --seeds 0 1 2
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
import torch

from poker_arena.encoding import ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker
from poker_arena.training import make_opponent
from poker_arena.training_b import load_checkpoint_b

from compare_agents import load_agent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hands", type=int, default=500,
                        help="deals to record states from (default: 500)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="seeds whose Agent B checkpoints to test")
    parser.add_argument("--opponent", default="raiser_loose",
                        help="scripted opponent during recording (default: raiser_loose)")
    parser.add_argument("--seed", type=int, default=10_000,
                        help="base deal seed; hand h uses seed + h")
    parser.add_argument("--checkpoint-dir",
                        default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--arm", default="history",
                        choices=["history", "control"],
                        help="which training arm's checkpoints to test")
    return parser.parse_args(argv)


def _record_states(net, encoder, env, hands, *, seed, opponent_name):
    """Play with history on and keep every decision state player 0 faced."""

    tracker = HistoryTracker()
    recorded: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    decisions = 0

    with torch.no_grad():
        for hand in range(hands):
            deal_seed = seed + hand
            env.seed(deal_seed)
            opponent = make_opponent(opponent_name, seed=deal_seed)

            state, player_id = reset_hand(env)
            small_blind = small_blind_player(state)
            tracker.start_hand(state)

            while not env.is_over():
                if player_id == 0:
                    encoded = encoder.encode(
                        state, is_small_blind=(small_blind == player_id)
                    )
                    sequence = tracker.tokens(viewer=player_id)
                    recorded.append(
                        (
                            encoded.features,
                            sequence.tokens,
                            sequence.mask,
                            encoded.legal_mask,
                        )
                    )
                    decisions += 1
                    logits, _ = net(
                        torch.from_numpy(encoded.features).unsqueeze(0),
                        torch.from_numpy(sequence.tokens).unsqueeze(0),
                        torch.from_numpy(sequence.mask).unsqueeze(0),
                        torch.from_numpy(encoded.legal_mask).unsqueeze(0),
                        use_history=True,
                    )
                    action = int(torch.argmax(logits, dim=-1))
                else:
                    action = int(opponent(state, player_id))

                tracker.record(state, player_id, env.actions[action])
                state, player_id = env.step(action)

    return recorded


def _counterfactual(net, recorded):
    """Return (fraction changed, mean |logit shift|) with history off vs on."""

    changed = 0
    shifts: list[float] = []
    with torch.no_grad():
        for features, tokens, mask, legal in recorded:
            batch = (
                torch.from_numpy(features).unsqueeze(0),
                torch.from_numpy(tokens).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(legal).unsqueeze(0),
            )
            logits_on, _ = net(*batch, use_history=True)
            logits_off, _ = net(*batch, use_history=False)
            if int(torch.argmax(logits_on, dim=-1)) != int(
                torch.argmax(logits_off, dim=-1)
            ):
                changed += 1
            shifts.append(float((logits_on - logits_off).abs().max(dim=-1).values))
    fraction = changed / len(recorded) if recorded else 0.0
    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    return fraction, mean_shift


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = make_environment(seed=args.seed)
    encoder = ObservationEncoder.from_environment(env)
    directory = Path(args.checkpoint_dir)

    print("Poker AI Arena - history reliance on a fixed state distribution")
    print()
    print(f"opponent      : {args.opponent}")
    print(f"deals         : {args.hands}   base seed {args.seed}")
    print(f"arm           : {args.arm}")
    print()
    print("                  decisions   action changes   mean |logit shift|")
    print("                  ---------   --------------   ------------------")

    for seed in args.seeds:
        name = (
            f"agent_b_seed{seed}_0050.pt"
            if args.arm == "history"
            else f"agent_b_control_seed{seed}_0050.pt"
        )
        path = directory / name
        if not path.is_file():
            print(f"  seed {seed}                 *missing* {name}")
            continue

        net, _ = load_checkpoint_b(path)
        net.eval()
        recorded = _record_states(
            net, encoder, env, args.hands, seed=args.seed,
            opponent_name=args.opponent,
        )
        fraction, mean_shift = _counterfactual(net, recorded)
        print(
            f"  seed {seed}         {len(recorded):>9}   {fraction:>12.3%}   "
            f"{mean_shift:>18.6f}"
        )

    print()
    print("The control arm is the baseline: its weights never saw the history, so")
    print("any change there is the network's untrained sensitivity to the channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
