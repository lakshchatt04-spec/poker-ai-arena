"""Break the features-only proxy and see whether the GRU still carries the mode.

Stage 3 found a training-time benefit for memory but almost no decision-time
reliance.  The pre-registered explanation is that the hidden fact -- "did the
agent make the first pre-flop raise?" -- is recoverable from the agent's own
features, because the greedy pre-flop opening is a deterministic function of its
hole cards and position.  A trunk can therefore learn a features-only proxy for
it, and the GRU becomes redundant at decision time.

This script tests that explanation without retraining.  It plays paired deals
under two opening policies:

``greedy``
    The agent plays its trained policy.  This reproduces the small ablation
    seen in ``compare_matrix.py`` and validates the protocol.

``forced``
    At the agent's *first* pre-flop decision, a per-hand seeded coin flip
    decides raise vs call/check.  The action is identical for both arms on the
    same deal, but it no longer depends on the agent's cards, so the features
    cannot reveal what the agent did.  Only the history channel still knows.

If the GRU holds real mode information that the proxy normally masks, then
zeroing it should hurt much more under ``forced`` than under ``greedy``.  If the
gap stays near zero, the trained network genuinely does not use memory at
decision time.

The comparison is a within-checkpoint ablation: the same weights, the same
deals, ``use_history`` toggled.  Both arms are off the training distribution in
the forced condition; that is the point of the counterfactual, and it is stated
in the report.

Usage::

    python scripts\\proxy_break.py --hands 2000
    python scripts\\proxy_break.py --opponent passive --hands 2000
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

from poker_arena.encoding import ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker, street_from_board_size
from poker_arena.opponents import notify_hand_start
from poker_arena.training import make_opponent
from poker_arena.training_b import AgentBPolicy, load_checkpoint_b

Z_95 = 1.959963984540054
CONDITIONS = ("greedy", "forced")

#: Rank order for the simple hole-card strength score.
RANKS = "23456789TJQKA"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hands", type=int, default=2_000,
        help="paired hands per seed and condition (default: 2000)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="training seeds of Agent B (default: 0 1 2)",
    )
    parser.add_argument(
        "--opponent", default="first_raiser",
        help="scripted opponent (default: first_raiser)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "first_raiser"),
        help="where agent_b_seed*_0050.pt lives",
    )
    parser.add_argument(
        "--seed-base", type=int, default=10_000,
        help="base deal seed; hand h uses seed + h (default: 10000)",
    )
    parser.add_argument(
        "--out", default=None,
        help="also write the report here as markdown",
    )
    return parser.parse_args(argv)


def _hole_strength(cards: list[str]) -> float:
    """Crude pre-flop strength: sum of the two ranks, normalised to [0, 1]."""

    values = sorted((RANKS.index(card[1]) + 2 for card in cards), reverse=True)
    return (values[0] + values[1]) / 28.0


def _forced_opening(state, rng: np.random.Generator) -> int:
    """Coin-flip between raising and calling/checking when both are legal."""

    legal = {int(action) for action in state["legal_actions"]}
    if 1 in legal and rng.random() < 0.5:
        return 1
    for action in (3, 0, 2):  # check, call, fold -- RLCard's decoded action
        if action in legal:
            return action
    return min(legal)


def _play_hand(
    env,
    policy: AgentBPolicy,
    opponent,
    tracker: HistoryTracker,
    *,
    forced: bool,
    deal_seed: int,
) -> tuple[float, dict[str, float]]:
    """Play one hand as player 0, optionally randomising its opening action."""

    state, player_id = reset_hand(env)
    small_blind = small_blind_player(state)
    tracker.start_hand(state)
    notify_hand_start(opponent, state)

    rng = np.random.default_rng(deal_seed * 2 + 1)
    agent_preflop_decisions = 0
    first_preflop_raiser: int | None = None
    opening_raise = 0.0
    greedy_raise = 0.0
    strength = 0.0
    recorded = False

    while not env.is_over():
        street = street_from_board_size(len(state["raw_obs"]["public_cards"]))
        if player_id == 0:
            greedy = int(policy(state, player_id, tracker, small_blind))
            action = greedy
            if street == 0 and agent_preflop_decisions == 0:
                if forced:
                    action = _forced_opening(state, rng)
                if not recorded:
                    opening_raise = float(env.actions[action] == "raise")
                    greedy_raise = float(env.actions[greedy] == "raise")
                    strength = _hole_strength(state["raw_obs"]["hand"])
                    recorded = True
            if street == 0:
                agent_preflop_decisions += 1
        else:
            action = int(opponent(state, player_id))
            if action not in state["legal_actions"]:
                raise ValueError(
                    f"opponent returned illegal action {action!r} for player "
                    f"{player_id}; legal: {sorted(state['legal_actions'])}"
                )

        if (
            street == 0
            and first_preflop_raiser is None
            and env.actions[action] == "raise"
        ):
            first_preflop_raiser = player_id

        tracker.record(state, player_id, env.actions[action])
        state, player_id = env.step(action)

    info = {
        "opening_raise": opening_raise,
        "greedy_raise": greedy_raise,
        "strength": strength,
        "agent_first_raiser": float(first_preflop_raiser == 0),
        "agent_acted_preflop": float(recorded),
    }
    return float(env.get_payoffs()[0]), info


def _run_condition(
    env,
    policy_history: AgentBPolicy,
    policy_zeroed: AgentBPolicy,
    hands: int,
    seed: int,
    opponent_name: str,
    *,
    forced: bool,
) -> dict[str, np.ndarray]:
    """Paired deals; history-on minus history-zeroed, plus opening stats."""

    tracker = HistoryTracker()
    payoffs_on = np.empty(int(hands), dtype=np.float64)
    payoffs_off = np.empty(int(hands), dtype=np.float64)
    stats = {
        key: np.empty(int(hands), dtype=np.float64)
        for key in (
            "opening_raise",
            "greedy_raise",
            "strength",
            "agent_first_raiser",
            "agent_acted_preflop",
        )
    }

    for hand in range(int(hands)):
        deal_seed = int(seed) + hand

        env.seed(deal_seed)
        opponent = make_opponent(opponent_name, seed=deal_seed)
        payoffs_on[hand], info = _play_hand(
            env, policy_history, opponent, tracker,
            forced=forced, deal_seed=deal_seed,
        )
        for key, value in info.items():
            stats[key][hand] = value

        env.seed(deal_seed)
        opponent = make_opponent(opponent_name, seed=deal_seed)
        payoffs_off[hand], _ = _play_hand(
            env, policy_zeroed, opponent, tracker,
            forced=forced, deal_seed=deal_seed,
        )

    return {
        "on": payoffs_on,
        "off": payoffs_off,
        "difference": payoffs_on - payoffs_off,
        **stats,
    }


def _quarter_raise_rates(
    strength: np.ndarray, opening_raise: np.ndarray
) -> tuple[float, float]:
    """Opening-raise rate in the weakest and strongest quarters of hands."""

    if strength.size < 8:
        return float("nan"), float("nan")
    low, high = np.quantile(strength, [0.25, 0.75])
    weak = opening_raise[strength <= low]
    strong = opening_raise[strength >= high]
    return float(weak.mean()), float(strong.mean())


def _summary(values: np.ndarray) -> tuple[float, float]:
    """Mean and two-sided 95 % half-width."""

    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean, Z_95 * stderr


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.hands < 2:
        raise SystemExit("--hands must be at least 2")

    directory = Path(args.checkpoint_dir)
    env = make_environment(seed=args.seed_base)
    encoder = ObservationEncoder.from_environment(env)

    lines: list[str] = []
    lines.append("Poker AI Arena - does memory matter when the proxy is broken?")
    lines.append("")
    lines.append(f"opponent      : {args.opponent}")
    lines.append(f"deals         : {args.hands} per seed and condition")
    lines.append(f"base seed     : {args.seed_base}")
    lines.append("")
    lines.append(
        "Each cell is history-on minus history-zeroed, same weights, same deals."
    )
    lines.append(
        "'forced' flips a per-hand coin for the agent's first pre-flop action, so"
    )
    lines.append("its cards no longer reveal what it did.")
    lines.append("")

    all_diffs = {condition: [] for condition in CONDITIONS}

    for seed in args.seeds:
        path = directory / f"agent_b_seed{seed}_0050.pt"
        if not path.is_file():
            lines.append(f"seed {seed}: missing checkpoint {path.name}")
            continue

        net, _ = load_checkpoint_b(path)
        net.eval()
        policy_on = AgentBPolicy(net, encoder, use_history=True)
        policy_off = AgentBPolicy(net, encoder, use_history=False)

        lines.append(f"## seed {seed}")
        lines.append("")

        for condition in CONDITIONS:
            forced = condition == "forced"
            data = _run_condition(
                env, policy_on, policy_off, args.hands, args.seed_base,
                args.opponent, forced=forced,
            )
            diff_mean, diff_half = _summary(data["difference"])
            on_mean, _ = _summary(data["on"])
            off_mean, _ = _summary(data["off"])
            weak, strong = _quarter_raise_rates(
                data["strength"], data["opening_raise"]
            )
            acted = data["agent_acted_preflop"] > 0
            raised = acted & (data["opening_raise"] > 0)
            first_raiser = data["agent_first_raiser"] > 0
            p_first_given_raise = (
                float(first_raiser[raised].mean()) if raised.any() else float("nan")
            )
            p_first_given_call = (
                float(first_raiser[acted & ~raised].mean())
                if (acted & ~raised).any() else float("nan")
            )
            all_diffs[condition].append(diff_mean)

            lines.append(f"### {condition}")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("|---|---|")
            lines.append(f"| history-on EV (bb/hand) | {on_mean:+.4f} |")
            lines.append(f"| history-zeroed EV (bb/hand) | {off_mean:+.4f} |")
            lines.append(
                f"| history-on minus zeroed | {diff_mean:+.4f} +/-{diff_half:.4f} |"
            )
            lines.append(
                f"| resolved | {'yes' if abs(diff_mean) > diff_half else 'no'} |"
            )
            lines.append(
                f"| opening raise rate, weak quarter | {weak:.1%} |"
            )
            lines.append(
                f"| opening raise rate, strong quarter | {strong:.1%} |"
            )
            lines.append(
                f"| P(agent first raiser | opened with raise) | "
                f"{p_first_given_raise:.1%} |"
            )
            lines.append(
                f"| P(agent first raiser | did not open raise) | "
                f"{p_first_given_call:.1%} |"
            )
            lines.append("")

    lines.append("## summary")
    lines.append("")
    lines.append("| condition | mean history-on minus zeroed | seeds |")
    lines.append("|---|---|---|")
    for condition in CONDITIONS:
        values = np.asarray(all_diffs[condition], dtype=np.float64)
        if values.size:
            lines.append(
                f"| {condition} | {values.mean():+.4f} | {values.size} |"
            )
        else:
            lines.append(f"| {condition} | n/a | 0 |")
    lines.append("")
    lines.append(
        "Reading: if 'forced' is materially larger than 'greedy', the GRU still"
    )
    lines.append(
        "carries the mode and the features-only proxy is what hides it at"
    )
    lines.append(
        "decision time. If the two are the same, the trained policy does not use"
    )
    lines.append("memory even when the proxy is unavailable.")

    report = "\n".join(lines)
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
