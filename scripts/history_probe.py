"""Does Agent A's observation already reveal the opponent's hidden mode?

Two opponents claim to hide something from the 131-feature observation.  This
script tests that claim before any training compute is spent.

`c_bettor` (``ContinuationBettor``)
    Raises pre-flop on a coin flip, then bets only if it was the pre-flop
    aggressor.  The hidden fact is "did *I* raise?".

`first_raiser` (``FirstRaiserBettor``)
    Raises pre-flop on a coin flip, then bets only if the *agent* made the
    first pre-flop raise.  The hidden fact is "who raised first?".

`coin_opener` (``CoinOpenerBettor``)
    A per-hand coin drawn from the opponent's own RNG selects an aggressive or
    passive line.  The hidden fact is the coin, which is independent of the
    agent's cards.  To keep it hidden from the observation, probe it with
    ``--agent complement``: the agent calls a bot raise and raises a bot call,
    so both coin values reach the flop with identical public features.

Protocol:

1. Play ``--hands`` deals with a fixed agent (uniform random by default) against
   the bot.  Record every post-flop decision the agent faced, together with the
   bot's true mode for that hand.
2. **Identifiability.**  Group the states by the betting summary alone -- chips,
   per-street raise counts, street, pot, amount to call and blind flag -- with
   the cards excluded, because cards are drawn independently of the mode.  If a
   summary appears with both modes, no function of the observation can separate
   them.  Report how many agent decisions fall in such mixed groups, how many of
   the *first* flop decisions are ambiguous, and how often the agent's first
   pre-flop response was complementary (call a bot raise, raise a bot call),
   because that is what keeps a private coin hidden for ``coin_opener``.
3. **Prediction.**  Fit a logistic classifier to predict the mode, splitting by
   hand so no deal leaks between train and test, and compare features-only
   against features + the 32x16 history token table.  Also report both on the
   mixed groups only.  There the observation is provably ambiguous, so any
   accuracy above chance must come from the history.  Two reference rows make
   that readable: the mixed groups' own majority baseline (a card-free model can
   never beat it, because every state in a group shares the same summary) and a
   cards-only model, which cannot identify the summary at all.  Cards are drawn
   independently of the mode, so cards-only at its baseline is the leakage test.
4. **Sanity check.**  Report how often the bot's first post-flop action is a bet
   or raise, split by mode.  If the two modes do not behave differently, the
   label is not worth predicting.

Reading: features-only near the majority baseline (and cards-only at the
mixed-group baseline) means the mode is genuinely hidden and the training
experiment is worth running.  A large features-only accuracy, or cards-only
above its baseline, means the bot reveals its mode through its own bets and
memory is redundant by construction.

Usage::

    python scripts\\history_probe.py --opponent first_raiser
    python scripts\\history_probe.py --opponent c_bettor --hands 4000
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from poker_arena.encoding import (
    FEATURE_DIM,
    RAISE_SLICE,
    ObservationEncoder,
    small_blind_player,
)
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker, street_from_board_size
from poker_arena.opponents import notify_hand_start
from poker_arena.training import make_opponent
from poker_arena.training_b import AgentBPolicy, load_checkpoint_b

#: The betting summary starts after the two 52-slot card blocks.
BETTING_SLICE = slice(RAISE_SLICE.start, FEATURE_DIM)

#: What each opponent is supposed to hide.
MODE_DESCRIPTIONS = {
    "c_bettor": "the bot raised pre-flop (it then continuation-bets)",
    "first_raiser": "the agent, not the bot, made the first pre-flop raise",
    "coin_opener": "the bot's per-hand coin: aggressive (1) or passive (0)",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hands", type=int, default=3_000,
        help="deals to play against the bot (default: 3000)",
    )
    parser.add_argument(
        "--seed", type=int, default=20_000,
        help="base deal seed; hand h uses seed + h (default: 20000)",
    )
    parser.add_argument(
        "--opponent", default="first_raiser",
        choices=sorted(MODE_DESCRIPTIONS),
        help="which hidden-mode opponent to probe (default: first_raiser)",
    )
    parser.add_argument(
        "--agent", default="random",
        choices=["random", "passive", "complement", "checkpoint"],
        help="how the probe agent plays (default: random legal action)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Agent B checkpoint to play greedily when --agent checkpoint",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.75,
        help="fraction of hands used for fitting (default: 0.75)",
    )
    parser.add_argument(
        "--epochs", type=int, default=400,
        help="full-batch logistic-regression epochs (default: 400)",
    )
    parser.add_argument(
        "--out", default=None,
        help="also write the report here as markdown",
    )
    return parser.parse_args(argv)


def _random_action(state, rng: np.random.Generator) -> int:
    return int(rng.choice(sorted(state["legal_actions"])))


def _passive_action(state) -> int:
    legal = state["legal_actions"]
    for action in (3, 0, 2):  # check, call, fold -- match RLCard's decoded action
        if action in legal:
            return action
    return int(min(legal))


def _complement_action(state, *, first_preflop_decision: bool) -> int:
    """Call a bot raise and raise a bot call, then play passively.

    Used with ``coin_opener`` to make the two coin values reach the flop with
    identical chips and raise counts.  The bot is the only other player, so a
    pre-flop raise before the agent acts must be the bot's.
    """

    legal = state["legal_actions"]
    if first_preflop_decision:
        raises = int((state.get("raw_obs") or {}).get("raise_nums", [0])[0])
        if raises >= 1:
            for action in (0, 3):  # call, else check
                if action in legal:
                    return action
        elif 1 in legal:
            return 1
    return _passive_action(state)


def _collect(
    hands: int, seed: int, opponent_name: str, agent_name: str,
    checkpoint: str | None = None,
) -> dict[str, np.ndarray]:
    """Play hands and record every post-flop decision the agent faced."""

    env = make_environment(seed=seed)
    encoder = ObservationEncoder.from_environment(env)
    rng = np.random.default_rng(seed + 104_729)
    agent_policy = None
    if checkpoint is not None:
        net, _ = load_checkpoint_b(checkpoint)
        net.eval()
        agent_policy = AgentBPolicy(net, encoder)

    features: list[np.ndarray] = []
    tokens: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    streets: list[int] = []
    hand_ids: list[int] = []
    first_flop_flags: list[bool] = []

    hand_labels: list[int | None] = [None] * int(hands)
    complement_labels: list[int] = [-1] * int(hands)
    first_postflop: list[tuple[int, int]] = []

    for hand in range(int(hands)):
        deal_seed = seed + hand
        env.seed(deal_seed)
        opponent = make_opponent(opponent_name, seed=deal_seed)
        state, player = reset_hand(env)
        small_blind = small_blind_player(state)
        tracker = HistoryTracker()
        tracker.start_hand(state)
        notify_hand_start(opponent, state)

        first_preflop_raiser: int | None = None
        bot_acted_preflop = False
        bot_first_preflop_raise = False
        first_aggressive: bool | None = None
        agent_preflop_decisions = 0
        agent_postflop_decisions = 0

        while not env.is_over():
            street = street_from_board_size(
                len(state["raw_obs"]["public_cards"])
            )

            if player == 0:
                if street > 0:
                    encoded = encoder.encode(
                        state, is_small_blind=(small_blind == 0)
                    )
                    sequence = tracker.tokens(viewer=0)
                    features.append(encoded.features.copy())
                    tokens.append(sequence.tokens.copy())
                    masks.append(sequence.mask.copy())
                    streets.append(street)
                    hand_ids.append(hand)
                    first_flop_flags.append(agent_postflop_decisions == 0)
                    agent_postflop_decisions += 1

                if agent_policy is not None:
                    action = int(
                        agent_policy(state, player, tracker, small_blind)
                    )
                elif agent_name == "random":
                    action = _random_action(state, rng)
                elif agent_name == "complement":
                    action = _complement_action(
                        state,
                        first_preflop_decision=(
                            street == 0 and agent_preflop_decisions == 0
                        ),
                    )
                else:
                    action = _passive_action(state)
                if street == 0:
                    if agent_preflop_decisions == 0 and bot_acted_preflop:
                        # Complementary means call a raise, raise a call/check.
                        if bot_first_preflop_raise:
                            complement_labels[hand] = int(
                                env.actions[action] != "raise"
                            )
                        else:
                            complement_labels[hand] = int(
                                env.actions[action] == "raise"
                            )
                    agent_preflop_decisions += 1
            else:
                action = int(opponent(state, player))
                if street == 0 and not bot_acted_preflop:
                    bot_acted_preflop = True
                    bot_first_preflop_raise = env.actions[action] == "raise"
                elif street > 0 and first_aggressive is None:
                    first_aggressive = env.actions[action] == "raise"

            if (
                street == 0
                and first_preflop_raiser is None
                and env.actions[action] == "raise"
            ):
                first_preflop_raiser = player

            tracker.record(state, player, env.actions[action])
            state, player = env.step(action)

        if opponent_name == "c_bettor":
            if not bot_acted_preflop:
                continue
            label = int(bot_first_preflop_raise)
        elif opponent_name == "coin_opener":
            coin = getattr(opponent, "coin", None)
            if coin is None:
                continue
            label = int(coin)
        else:
            label = int(first_preflop_raiser == 0)

        hand_labels[hand] = label
        if first_aggressive is not None:
            first_postflop.append((label, int(first_aggressive)))

    labels = np.asarray(
        [hand_labels[hand] for hand in hand_ids], dtype=np.int64
    )

    return {
        "features": np.stack(features) if features else np.empty((0, FEATURE_DIM)),
        "tokens": np.stack(tokens) if tokens else np.empty((0, 32, 16)),
        "masks": np.stack(masks) if masks else np.empty((0, 32)),
        "labels": labels,
        "streets": np.asarray(streets, dtype=np.int64),
        "hand_ids": np.asarray(hand_ids, dtype=np.int64),
        "first_flop": np.asarray(first_flop_flags, dtype=bool),
        "complement": np.asarray(complement_labels, dtype=np.int64),
        "first_postflop": np.asarray(first_postflop, dtype=np.int64).reshape(-1, 2),
    }


def _summary_keys(features: np.ndarray) -> list[bytes]:
    return [np.round(vector[BETTING_SLICE], 6).tobytes() for vector in features]


def _identifiability(
    features: np.ndarray, labels: np.ndarray, keys: list[bytes]
) -> tuple[dict[str, int], set[bytes]]:
    """Group states by the card-free betting summary and count mixed groups."""

    groups: dict[bytes, list[int]] = defaultdict(lambda: [0, 0])
    for key, label in zip(keys, labels):
        groups[key][int(label)] += 1

    mixed = {
        key: counts for key, counts in groups.items() if counts[0] and counts[1]
    }
    return (
        {
            "decisions": int(labels.size),
            "summaries": len(groups),
            "mixed_summaries": len(mixed),
            "mixed_decisions": int(
                sum(sum(counts) for counts in mixed.values())
            ),
        },
        set(mixed),
    )


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 400,
    learning_rate: float = 0.05,
) -> float:
    """Fit a standardised logistic model and return its test accuracy."""

    if x_train.size == 0 or x_test.size == 0 or y_train.size == 0:
        return float("nan")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0

    xt = torch.from_numpy(((x_train - mean) / std).astype(np.float32))
    xs = torch.from_numpy(((x_test - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    ys = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

    torch.manual_seed(seed)
    model = torch.nn.Linear(xt.shape[1], 1)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(int(epochs)):
        optimiser.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        predictions = (model(xs) > 0).float()
    return float((predictions == ys).float().mean())


def _report(args: argparse.Namespace, data: dict[str, np.ndarray]) -> str:
    features = data["features"]
    labels = data["labels"]
    lines: list[str] = []

    lines.append("Poker AI Arena - is the opponent's hidden mode visible?")
    lines.append("")
    lines.append(f"opponent      : {args.opponent}")
    lines.append(f"hidden mode   : {MODE_DESCRIPTIONS[args.opponent]}")
    lines.append(f"probe agent   : {args.agent}")
    if getattr(args, "checkpoint", None):
        lines.append(f"checkpoint    : {args.checkpoint}")
    lines.append(f"hands         : {args.hands}   base seed {args.seed}")
    lines.append(f"post-flop decision states: {labels.size}")
    if labels.size == 0:
        lines.append("")
        lines.append("No post-flop decisions were recorded; nothing to test.")
        return "\n".join(lines)

    lines.append(
        f"mode balance  : {labels.mean():.1%} of decisions were mode 1"
    )
    lines.append("")

    keys = _summary_keys(features)
    ident, mixed_keys = _identifiability(features, labels, keys)
    mixed_fraction = ident["mixed_decisions"] / ident["decisions"]
    lines.append("1. Identifiability from the betting summary (cards excluded)")
    lines.append(f"   decisions                : {ident['decisions']}")
    lines.append(f"   distinct summaries       : {ident['summaries']}")
    lines.append(
        f"   summaries seen in 2 modes: {ident['mixed_summaries']}"
    )
    lines.append(
        f"   decisions in mixed groups: {ident['mixed_decisions']} "
        f"({mixed_fraction:.1%})"
    )
    lines.append(
        "   the rest sit in groups whose summary already determines the mode"
    )
    first_flop = data["first_flop"]
    if first_flop.any():
        mixed_first = first_flop & np.array(
            [key in mixed_keys for key in keys], dtype=bool
        )
        lines.append(
            f"   first flop decisions mixed: "
            f"{int(mixed_first.sum())} of {int(first_flop.sum())} "
            f"({mixed_first.sum() / first_flop.sum():.1%})"
        )
    complement = data["complement"]
    complement = complement[complement >= 0]
    if complement.size:
        lines.append(
            f"   agent's first pre-flop response complementary: "
            f"{complement.mean():.1%} of {complement.size} bot-opened hands"
        )
    lines.append("")

    cutoff = int(args.hands * args.train_fraction)
    train_mask = data["hand_ids"] < cutoff
    test_mask = ~train_mask
    if not test_mask.any():
        lines.append("Not enough hands for a train/test split.")
        return "\n".join(lines)

    majority = max(
        float(labels[test_mask].mean()), 1.0 - float(labels[test_mask].mean())
    )

    x_base = features
    x_cards = features[:, :RAISE_SLICE.start]
    x_history = np.concatenate(
        [
            features,
            data["tokens"].reshape(data["tokens"].shape[0], -1),
            data["masks"].astype(np.float32),
        ],
        axis=1,
    )
    x_tokens = np.concatenate(
        [
            data["tokens"].reshape(data["tokens"].shape[0], -1),
            data["masks"].astype(np.float32),
        ],
        axis=1,
    )

    accuracy_tokens = _fit_logistic(
        x_tokens[train_mask], labels[train_mask],
        x_tokens[test_mask], labels[test_mask],
        epochs=args.epochs,
    )
    accuracy_base = _fit_logistic(
        x_base[train_mask], labels[train_mask],
        x_base[test_mask], labels[test_mask],
        epochs=args.epochs,
    )
    accuracy_history = _fit_logistic(
        x_history[train_mask], labels[train_mask],
        x_history[test_mask], labels[test_mask],
        epochs=args.epochs,
    )

    lines.append("2. Predicting the mode from the agent's own decision states")
    lines.append(f"   majority baseline        : {majority:.1%}")
    lines.append(f"   features only            : {accuracy_base:.1%}")
    lines.append(f"   history tokens only      : {accuracy_tokens:.1%}")
    lines.append(f"   features + history       : {accuracy_history:.1%}")
    lines.append(
        "   (logistic regression, split by hand, cards included in 'features')"
    )
    lines.append("")

    mixed_test = test_mask & np.array(
        [key in mixed_keys for key in keys], dtype=bool
    )
    if mixed_test.any():
        mixed_labels = labels[mixed_test]
        mixed_majority = max(
            float(mixed_labels.mean()), 1.0 - float(mixed_labels.mean())
        )
        base_mixed = _fit_logistic(
            x_base[train_mask], labels[train_mask],
            x_base[mixed_test], labels[mixed_test],
            epochs=args.epochs,
        )
        history_mixed = _fit_logistic(
            x_history[train_mask], labels[train_mask],
            x_history[mixed_test], labels[mixed_test],
            epochs=args.epochs,
        )
        # A mixed group shares every non-card feature, so a card-free classifier
        # can only return one constant prediction per summary.  It therefore
        # scores at most that summary's label imbalance, even with no leak.
        # Cards-only is the real leak test: cards cannot identify the summary.
        cards_mixed = _fit_logistic(
            x_cards[train_mask], labels[train_mask],
            x_cards[mixed_test], labels[mixed_test],
            epochs=args.epochs,
        )
        lines.append(
            f"   on mixed groups only     : {base_mixed:.1%} features / "
            f"{history_mixed:.1%} features + history "
            f"(n={int(mixed_test.sum())})"
        )
        lines.append(
            f"   mixed-group majority     : {mixed_majority:.1%} "
            f"(a card-free model cannot beat this on a shared summary)"
        )
        lines.append(
            f"   mixed groups, cards only : {cards_mixed:.1%}"
        )
        lines.append("")

    first = data["first_postflop"]
    if first.size:
        lines.append("3. Sanity check: does the mode predict the opponent's line?")
        for mode in (0, 1):
            subset = first[first[:, 0] == mode]
            if subset.size:
                rate = float(subset[:, 1].mean())
                lines.append(
                    f"   mode {mode}: first post-flop action was a bet/raise "
                    f"in {rate:.1%} of {subset.shape[0]} hands"
                )
        lines.append("")

    lines.append("Reading: features-only near the majority baseline (and")
    lines.append("cards-only at the mixed-group baseline) means the mode is")
    lines.append("genuinely hidden; a large features-only accuracy, or")
    lines.append("cards-only above its baseline, means the bot reveals its")
    lines.append("mode through its own bets and memory is redundant.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.hands < 2:
        raise SystemExit("--hands must be at least 2")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be between 0 and 1")
    if args.agent == "checkpoint" and not args.checkpoint:
        raise SystemExit("--agent checkpoint needs --checkpoint PATH")

    data = _collect(
        args.hands, args.seed, args.opponent, args.agent,
        checkpoint=args.checkpoint if args.agent == "checkpoint" else None,
    )
    report = _report(args, data)
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
