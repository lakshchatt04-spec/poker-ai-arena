"""Measure whether the auxiliary head recovered the opponent's private coin.

The auxiliary-head experiment trains Agent B with one extra supervised loss:
``BCE(aux_head(GRU state), opponent coin)``.  The head reads **only** the
recurrent state, never the observation trunk, so a high accuracy is evidence
that the GRU itself carries the coin rather than the hole cards.

This script answers the measurement question after training:

* play fresh deals against ``coin_opener`` with the checkpoint's greedy policy;
* record every decision the learner faced, together with the hand's true coin;
* score those decisions with the trained auxiliary head, once with the history
  channel on and once with it zeroed.

The history-off row is the control.  With the GRU output zeroed the auxiliary
head receives a constant input, so it cannot rank the coins: its accuracy stays
at the majority class and its AUC at 0.5 by construction.  A history-on accuracy
above that is the head using the recurrent channel.

The same decisions are split by history length.  The first decision of a hand
has no tokens, so any accuracy there is impossible; post-flop decisions carry
the opponent's pre-flop action and are where the coin should become readable.

Usage::

    python scripts\\aux_coin_probe.py --checkpoint checkpoints\\auxcoin100\\agent_b_auxcoin_seed0_0100.pt
    python scripts\\aux_coin_probe.py --checkpoint ... --hands 3000 --out runs\\auxcoin100\\probe_seed0.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from poker_arena.encoding import ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker
from poker_arena.opponents import notify_hand_start
from poker_arena.training import make_opponent
from poker_arena.training_b import AgentBPolicy, load_checkpoint_b

#: Normal 95 % half-width multiplier.
Z_95 = 1.959963984540054


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", required=True,
        help="Agent B checkpoint with an auxiliary coin head",
    )
    parser.add_argument(
        "--hands", type=int, default=2_000,
        help="fresh deals against coin_opener (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=30_000,
        help="base deal seed; hand h uses seed + h (default: 30000)",
    )
    parser.add_argument(
        "--opponent", default="coin_opener",
        help="opponent whose coin is the label (default: coin_opener)",
    )
    parser.add_argument(
        "--out", default=None,
        help="also write the report here as markdown",
    )
    return parser.parse_args(argv)


def _collect(
    checkpoint: str, hands: int, seed: int, opponent_name: str
) -> dict[str, np.ndarray]:
    """Play deals and record the learner's decision states and true coins."""

    net, metadata = load_checkpoint_b(checkpoint)
    if not net.aux_coin:
        raise SystemExit(
            f"{checkpoint} has no auxiliary coin head; train with --aux-coin"
        )
    net.eval()

    env = make_environment(seed=seed)
    encoder = ObservationEncoder.from_environment(env)
    trained_with_history = bool(
        dict(metadata.get("config", {})).get("use_history", True)
    )
    policy = AgentBPolicy(net, encoder, use_history=trained_with_history)

    features: list[np.ndarray] = []
    tokens: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    legal: list[np.ndarray] = []
    labels: list[int] = []
    lengths: list[int] = []

    for hand in range(int(hands)):
        deal_seed = seed + hand
        env.seed(deal_seed)
        opponent = make_opponent(opponent_name, seed=deal_seed)
        state, player = reset_hand(env)
        small_blind = small_blind_player(state)
        tracker = HistoryTracker()
        tracker.start_hand(state)
        notify_hand_start(opponent, state)
        coin = getattr(opponent, "coin", None)
        if coin is None:
            raise SystemExit(
                f"opponent {opponent_name!r} did not expose a per-hand coin"
            )
        label = int(bool(coin))

        while not env.is_over():
            sequence = tracker.tokens(viewer=player)
            if player == 0:
                encoded = encoder.encode(
                    state, is_small_blind=(small_blind == player)
                )
                features.append(encoded.features.copy())
                tokens.append(sequence.tokens.copy())
                masks.append(sequence.mask.copy())
                legal.append(encoded.legal_mask.copy())
                labels.append(label)
                lengths.append(int(sequence.mask.sum()))
                action = int(policy(state, player, tracker, small_blind))
            else:
                action = int(opponent(state, player))

            tracker.record(state, player, env.actions[action])
            state, player = env.step(action)

    return {
        "features": np.stack(features),
        "tokens": np.stack(tokens),
        "masks": np.stack(masks),
        "legal": np.stack(legal),
        "labels": np.asarray(labels, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }


def _score(net, data: dict[str, np.ndarray], *, use_history: bool) -> np.ndarray:
    """Auxiliary logits for every recorded decision, one batched forward pass."""

    with torch.no_grad():
        _, _, aux = net.forward_with_aux(
            torch.from_numpy(data["features"]),
            torch.from_numpy(data["tokens"]),
            torch.from_numpy(data["masks"]),
            torch.from_numpy(data["legal"]),
            use_history=use_history,
        )
    return aux.numpy().astype(np.float64)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""

    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1
    n_one = int(positive.sum())
    n_zero = int(labels.size - n_one)
    if n_one == 0 or n_zero == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered = np.asarray(scores)[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < ordered.size:
        end = start
        while end + 1 < ordered.size and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1

    return float(
        (ranks[positive].sum() - n_one * (n_one + 1) / 2.0) / (n_one * n_zero)
    )


def _accuracy_row(scores: np.ndarray, labels: np.ndarray) -> tuple[int, float, float, float]:
    predicted = (scores > 0.0).astype(np.int64)
    accuracy = float((predicted == labels).mean())
    half = Z_95 * float(np.sqrt(accuracy * (1.0 - accuracy) / labels.size))
    majority = max(float(labels.mean()), 1.0 - float(labels.mean()))
    return int(labels.size), accuracy, half, majority


def _report(
    args: argparse.Namespace, data: dict[str, np.ndarray], net
) -> str:
    labels = data["labels"]
    lines: list[str] = []
    lines.append("Poker AI Arena - does the auxiliary head recover the coin?")
    lines.append("")
    lines.append(f"checkpoint : {args.checkpoint}")
    lines.append(f"opponent   : {args.opponent}")
    lines.append(f"hands      : {args.hands}   base seed {args.seed}")
    lines.append(f"decisions  : {labels.size}")
    lines.append(f"coin balance: {labels.mean():.1%} of decisions were coin 1")
    lines.append("")

    rows = []
    for name, use_history in (("history on", True), ("history off", False)):
        scores = _score(net, data, use_history=use_history)
        sample, accuracy, half, majority = _accuracy_row(scores, labels)
        rows.append(
            (name, sample, accuracy, half, majority, _auc(scores, labels))
        )

    lines.append("1. Auxiliary-head coin prediction on fresh deals")
    lines.append("   | channel | decisions | accuracy | 95% | majority | AUC |")
    lines.append("   |---|---|---|---|---|---|")
    for name, sample, accuracy, half, majority, auc in rows:
        lines.append(
            f"   | {name} | {sample} | {accuracy:.1%} | "
            f"+/-{half:.1%} | {majority:.1%} | {auc:.3f} |"
        )
    lines.append("")
    lines.append(
        "   history off feeds the head a zero vector, so it can only return a "
        "constant:"
    )
    lines.append(
        "   its AUC is 0.5 and its accuracy is that constant class's rate."
    )
    lines.append("")

    scores = _score(net, data, use_history=True)
    first = data["lengths"] == 0
    later = ~first
    lines.append("2. Split by whether any history exists yet")
    lines.append("   | subset | decisions | accuracy | AUC |")
    lines.append("   |---|---|---|---|")
    for name, mask in (
        ("first decision (empty history)", first),
        ("later decisions", later),
        ("history length >= 3", data["lengths"] >= 3),
    ):
        if not mask.any():
            lines.append(f"   | {name} | 0 | n/a | n/a |")
            continue
        sample, accuracy, _, _ = _accuracy_row(scores[mask], labels[mask])
        lines.append(
            f"   | {name} | {sample} | {accuracy:.1%} | "
            f"{_auc(scores[mask], labels[mask]):.3f} |"
        )
    lines.append("")
    lines.append("Reading: an AUC near 0.5 on the history-on row means the GRU")
    lines.append("state does not carry the coin even when a supervised head asks")
    lines.append("for it; a high AUC there means the head can read the coin but")
    lines.append("the policy may still ignore it. Pair this with the")
    lines.append("history-vs-control and within-checkpoint ablation tables to")
    lines.append("separate 'the channel carries the coin' from 'the policy uses")
    lines.append("it'.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.hands < 2:
        raise SystemExit("--hands must be at least 2")

    net, _ = load_checkpoint_b(args.checkpoint)
    data = _collect(args.checkpoint, args.hands, args.seed, args.opponent)
    report = _report(args, data, net)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# " + report + "\n", encoding="utf-8")
        print()
        print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
