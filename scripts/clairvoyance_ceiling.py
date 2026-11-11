"""How much is perfect knowledge of the private coin worth at the first flop?

The Stage 3c result is that a hidden per-hand coin is worth `+0.204` bb/hand
while training but only `+0.015` when the same weights are replayed with the
history channel zeroed.  One reading is that the channel is an optimisation
scaffold; the other is that the policy simply has not learned to use it.  This
script bounds which reading is possible.

At the agent's first flop decision against ``CoinOpenerBettor`` the coin is
still hidden from the 131 features: the two pre-flop lines are built to reach
the flop with the same chips, raise counts and board.  It is, however, visible
in the action record, which is exactly what Agent B's history channel carries.
A clairvoyant player would know the coin outright at that decision.

The counterfactual is a *fork*:

1. Replay a deal to the agent's first flop decision and snapshot the whole game
   (RLCard's ``allow_step_back`` stack), the history tracker, the deck order and
   the per-street raise counts.
2. For every legal action and every sampled turn/river runout, apply the action,
   then let the deployed policy play the rest of the hand.  Step back to the fork
   after each playout.
3. Compare the value of the best action *given the coin* with the value of the
   action the deployed policy actually chose.

Step 2 uses the same runouts for every action, so the comparison is paired and
the difference is not runout noise.  Two estimators bracket the ceiling:

* **cross-fitted** -- choose the best action on half the runouts, evaluate it on
  the other half, and average over both directions.  This is a lower bound: the
  selector is noisier than a true clairvoyant.
* **in-sample** -- choose and evaluate on all runouts.  This is an upper bound,
  because a finite runout sample makes the maximum look better than it is.

The truth lies between them.  If both are of order the measured ablation
(`+0.015` bb/hand), no memory design of this shape can gain much by recovering
the coin, and the training-time benefit really is an optimisation effect.

Only fork points whose card-free betting summary appears with both coins are
used.  Elsewhere the features already determine the coin, so a clairvoyant has
no extra information and including those states would dilute the estimate.

The per-street raise count needs care.  RLCard's ``step_back`` restores
``history_raises_nums`` (plural), but the game reads ``history_raise_nums``
(singular), so the restored observation's raise block is stale.  ``_restore``
rewrites the attribute and asserts the observation matches the snapshot.

Usage::

    python scripts\\clairvoyance_ceiling.py --hands 300 --runouts 16
    python scripts\\clairvoyance_ceiling.py --hands 1000 --seeds 0 1 2 ^
        --history-prefix agent_b_coin --out runs/clairvoyance_ceiling.md
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch

from poker_arena.encoding import (
    FEATURE_DIM,
    RAISE_SLICE,
    ObservationEncoder,
    small_blind_player,
)
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker
from poker_arena.opponents import notify_hand_start
from poker_arena.training import Z_95, make_opponent
from poker_arena.training_b import AgentBPolicy, load_checkpoint_b

#: The card-free part of the encoder's features: chips, raise counts, street,
#: pot, amount to call and the blind flag.  Two states with the same key cannot
#: be told apart without the cards, which are drawn independently of the coin.
BETTING_SLICE = slice(RAISE_SLICE.start, FEATURE_DIM)

Policy = Callable[[Any, int, HistoryTracker, int], int]


@dataclass(frozen=True)
class FirstFlopRecord:
    """One deal that reached the agent's first flop decision."""

    hand: int
    deal_seed: int
    flat_key: bytes
    coin: int
    payoff: float


@dataclass(frozen=True)
class ForkSnapshot:
    """Everything needed to return to the agent's first flop decision."""

    hand: int
    deal_seed: int
    coin: int
    steps: int
    small_blind: int
    obs: np.ndarray
    legal_actions: tuple[int, ...]
    raise_nums: tuple[int, ...]
    action_recorder_length: int
    tracker: HistoryTracker
    deck: tuple[Any, ...]
    natural_runout: tuple[str, str]


@dataclass(frozen=True)
class ForkResult:
    """The forked values for one first-flop decision."""

    hand: int
    coin: int
    actual_action: int
    best_action_in_sample: int
    legal_actions: tuple[int, ...]
    q_in_sample: tuple[float, ...]
    q_natural: tuple[float, ...]
    cross_fitted_ceiling: float
    in_sample_ceiling: float
    per_runout_ceiling: float
    actual_q_coin: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hands", type=int, default=1000,
                        help="deals to play per checkpoint (default: 1000)")
    parser.add_argument("--seed", type=int, default=30_000,
                        help="base deal seed; deal h uses seed + h (default: 30000)")
    parser.add_argument("--opponent", default="coin_opener",
                        help="scripted opponent with a per-hand private coin "
                             "(default: coin_opener)")
    parser.add_argument("--runouts", type=int, default=32,
                        help="sampled turn/river pairs per action (default: 32)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="checkpoint seeds to evaluate (default: 0 1 2)")
    parser.add_argument("--arm", default="history", choices=["history", "control"],
                        help="which training arm's checkpoints to test")
    parser.add_argument("--history-prefix", default="agent_b_coin",
                        help="run-name prefix of the history arm")
    parser.add_argument("--control-prefix", default="agent_b_coin_control",
                        help="run-name prefix of the control arm")
    parser.add_argument("--iteration", type=int, default=50,
                        help="checkpoint iteration to test (default: 50)")
    parser.add_argument("--checkpoint-dir",
                        default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--out", default=None,
                        help="also write the report here as markdown")
    return parser.parse_args(argv)


def _betting_key(features: np.ndarray) -> bytes:
    """Card-free key for a decision state."""

    return np.round(np.asarray(features)[BETTING_SLICE], 6).tobytes()


def _action(
    policy: Policy,
    state: dict[str, Any],
    tracker: HistoryTracker,
    small_blind: int,
) -> int:
    action = int(policy(state, 0, tracker, small_blind))
    if action not in state["legal_actions"]:
        raise ValueError(
            f"policy returned illegal action {action!r}; "
            f"legal: {sorted(state['legal_actions'])}"
        )
    return action


def _opponent_action(opponent: Any, state: dict[str, Any], player: int) -> int:
    action = int(opponent(state, player))
    if action not in state["legal_actions"]:
        raise ValueError(
            f"opponent returned illegal action {action!r}; "
            f"legal: {sorted(state['legal_actions'])}"
        )
    return action


def _is_first_flop_decision(
    state: dict[str, Any], player: int, recorded: bool
) -> bool:
    return (
        not recorded
        and player == 0
        and len(state["raw_obs"]["public_cards"]) == 3
    )


def play_to_first_flop(
    env: Any,
    policy: Policy,
    opponent: Any,
    tracker: HistoryTracker,
    deal_seed: int,
) -> tuple[dict[str, Any], int, int, int] | None:
    """Reset one seeded deal and stop at the agent's first flop decision.

    Returns ``(state, player, small_blind, steps)`` or ``None`` when the hand
    ends before the flop.  ``steps`` is ``len(env.game.history)`` at the fork
    when the environment allows step-back, so it doubles as the restore depth.
    """

    env.seed(deal_seed)
    state, player = reset_hand(env)
    small_blind = small_blind_player(state)
    tracker.start_hand(state)
    notify_hand_start(opponent, state)

    while not env.is_over():
        if _is_first_flop_decision(state, player, recorded=False):
            return state, player, small_blind, len(env.game.history)
        if player == 0:
            action = _action(policy, state, tracker, small_blind)
        else:
            action = _opponent_action(opponent, state, player)
        tracker.record(state, player, env.actions[action])
        state, player = env.step(action)

    return None


def play_out(
    env: Any,
    policy: Policy,
    opponent: Any,
    tracker: HistoryTracker,
    state: dict[str, Any],
    player: int,
    small_blind: int,
) -> float:
    """Finish the hand and return player 0's payoff in big blinds."""

    while not env.is_over():
        if player == 0:
            action = _action(policy, state, tracker, small_blind)
        else:
            action = _opponent_action(opponent, state, player)
        tracker.record(state, player, env.actions[action])
        state, player = env.step(action)
    return float(env.get_payoffs()[0])


def make_snapshot(
    env: Any,
    state: dict[str, Any],
    tracker: HistoryTracker,
    *,
    hand: int,
    deal_seed: int,
    coin: int,
    steps: int,
    small_blind: int,
) -> ForkSnapshot:
    """Freeze the fork point, including RLCard state outside the observation."""

    deck = tuple(env.game.dealer.deck)
    if len(deck) < 2:
        raise RuntimeError("expected at least two cards left at the flop")
    return ForkSnapshot(
        hand=hand,
        deal_seed=deal_seed,
        coin=coin,
        steps=steps,
        small_blind=small_blind,
        obs=state["obs"].copy(),
        legal_actions=tuple(sorted(int(a) for a in state["legal_actions"])),
        raise_nums=tuple(int(value) for value in env.game.history_raise_nums),
        action_recorder_length=len(env.action_recorder),
        tracker=copy.deepcopy(tracker),
        deck=deck,
        natural_runout=(str(deck[-1].get_index()), str(deck[-2].get_index())),
    )


def restore(env: Any, snapshot: ForkSnapshot) -> dict[str, Any]:
    """Step back to ``snapshot`` and repair the raise-count observation.

    RLCard's ``LimitHoldemGame.step_back`` assigns the snapshot's raise counts
    to ``history_raises_nums`` while ``get_state`` reads ``history_raise_nums``,
    so the native restore leaves the raise block frozen at the post-fork values.
    Rewriting the attribute is what makes the restored state equal to the one the
    policy actually saw.
    """

    forward = len(env.game.history) - snapshot.steps
    if forward < 0:
        raise RuntimeError("the fork point is not on the environment's history")
    for _ in range(forward):
        if not env.step_back():
            raise RuntimeError("step_back failed before reaching the fork point")

    env.game.history_raise_nums = list(snapshot.raise_nums)
    del env.action_recorder[snapshot.action_recorder_length:]

    restored = env.get_state(0)
    if not np.array_equal(restored["obs"], snapshot.obs):
        differing = np.flatnonzero(restored["obs"] != snapshot.obs)
        raise RuntimeError(
            f"restored observation differs from the snapshot at {differing.tolist()}"
        )
    if tuple(sorted(int(a) for a in restored["legal_actions"])) != snapshot.legal_actions:
        raise RuntimeError("restored legal actions differ from the snapshot")
    if int(env.game.get_player_id()) != 0:
        raise RuntimeError("restored player is not the agent")
    return restored


def deck_with_runout(
    snapshot: ForkSnapshot, turn: str, river: str
) -> list[Any]:
    """Order the deck so the next two deals are ``turn`` then ``river``.

    RLCard card ids are strings such as ``"SA"`` (``Card.get_index``).  The
    dealer pops from the end, so the turn goes last and the river second to
    last.  The rest of the deck keeps its order; only two cards are dealt after
    the flop.
    """

    by_index = {str(card.get_index()): card for card in snapshot.deck}
    if turn not in by_index or river not in by_index:
        raise ValueError(f"runout {(turn, river)} is not in the remaining deck")
    if turn == river:
        raise ValueError("turn and river must be different cards")
    rest = [
        card for card in snapshot.deck
        if str(card.get_index()) not in (turn, river)
    ]
    return rest + [by_index[river], by_index[turn]]


def sample_runouts(
    snapshot: ForkSnapshot, count: int, rng: np.random.Generator
) -> tuple[tuple[str, str], ...]:
    """Sample ``count`` distinct ordered turn/river pairs from the live deck."""

    if count < 2:
        raise ValueError(f"runouts must be at least 2, got {count!r}")
    indices = tuple(str(card.get_index()) for card in snapshot.deck)
    if count > len(indices) * (len(indices) - 1):
        raise ValueError("more runouts requested than distinct ordered pairs")

    seen: set[tuple[str, str]] = set()
    runouts: list[tuple[str, str]] = []
    while len(runouts) < count:
        pair = tuple(
            str(value)
            for value in rng.choice(indices, size=2, replace=False)
        )
        if pair in seen:
            continue
        seen.add(pair)
        runouts.append(pair)
    return tuple(runouts)


@dataclass(frozen=True)
class CeilingEstimates:
    """Value available from picking a different first-flop action."""

    cross_fitted: float
    in_sample: float
    per_runout: float
    best_row: int
    q: tuple[float, ...]


def ceiling_estimates(values: np.ndarray, actual_row: int) -> CeilingEstimates:
    """Bracket the per-decision ceiling from a ``(actions, runouts)`` value table.

    ``values[a, r]`` is the payoff of taking action ``a`` and then playing the
    deployed policy when runout ``r`` appears.  Every column is a common random
    number across actions, so differences are paired.

    * ``cross_fitted`` -- pick the best action on one half of the runouts and
      score it on the other half, then average over both directions.  The
      selector never sees the runouts it is scored on.
    * ``in_sample`` -- pick and score on every runout.  The maximum of a finite
      sample is biased upward, so this is an upper bound.
    * ``per_runout`` -- the best action *per runout*, averaged.  This includes
      foresight of the runout and is therefore not achievable at decision time;
      it is only a scale reference.

    All three are measured against the action actually taken, so a policy that
    already chooses optimally scores zero.
    """

    table = np.asarray(values, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(
            f"need at least two runout columns, got shape {table.shape!r}"
        )
    if not 0 <= actual_row < table.shape[0]:
        raise ValueError(
            f"actual_row {actual_row!r} is not an action row of {table.shape[0]}"
        )

    q = table.mean(axis=1)
    best_row = int(np.argmax(q))

    half = table.shape[1] // 2
    differences: list[float] = []
    for selection, evaluation in (
        (slice(0, half), slice(half, None)),
        (slice(half, None), slice(0, half)),
    ):
        selected = int(np.argmax(table[:, selection].mean(axis=1)))
        differences.append(
            float(
                table[selected, evaluation].mean()
                - table[actual_row, evaluation].mean()
            )
        )
    cross_fitted = float(np.mean(differences))

    in_sample = float(q[best_row] - q[actual_row])
    per_runout = float(np.mean(table.max(axis=0) - table[actual_row, :]))

    return CeilingEstimates(
        cross_fitted=cross_fitted,
        in_sample=in_sample,
        per_runout=per_runout,
        best_row=best_row,
        q=tuple(float(value) for value in q),
    )


def evaluate_fork(
    env: Any,
    policy: Policy,
    opponent: Any,
    snapshot: ForkSnapshot,
    state: dict[str, Any],
    runouts: Sequence[tuple[str, str]],
) -> ForkResult:
    """Value every legal action on the same runouts, then restore the fork."""

    actual = _action(policy, state, snapshot.tracker, snapshot.small_blind)
    natural_turn, natural_river = snapshot.natural_runout
    # Natural runout first, then the sampled set.  The natural playout of the
    # actual action is the deployed hand, which is the sanity check in the tests.
    all_runouts = ((natural_turn, natural_river), *runouts)
    values = np.empty((len(snapshot.legal_actions), len(all_runouts)), dtype=np.float64)

    for row, action in enumerate(snapshot.legal_actions):
        for column, (turn, river) in enumerate(all_runouts):
            # The forked action is part of the record the rest of the hand sees.
            # ``play_out`` only records actions *after* the fork, so without this
            # line the replay's GRU history is one token shorter than the actual
            # hand's and the policy silently diverges at the next decision.
            tracker = copy.deepcopy(snapshot.tracker)
            tracker.record(state, 0, env.actions[action])
            env.game.dealer.deck = deck_with_runout(snapshot, turn, river)
            next_state, next_player = env.step(action)
            values[row, column] = play_out(
                env, policy, opponent, tracker, next_state, next_player,
                snapshot.small_blind,
            )
            restore(env, snapshot)

    natural_values = values[:, 0]
    actual_row = snapshot.legal_actions.index(actual)
    estimates = ceiling_estimates(values[:, 1:], actual_row)

    return ForkResult(
        hand=snapshot.hand,
        coin=snapshot.coin,
        actual_action=actual,
        best_action_in_sample=int(snapshot.legal_actions[estimates.best_row]),
        legal_actions=snapshot.legal_actions,
        q_in_sample=estimates.q,
        q_natural=tuple(float(value) for value in natural_values),
        cross_fitted_ceiling=estimates.cross_fitted,
        in_sample_ceiling=estimates.in_sample,
        per_runout_ceiling=estimates.per_runout,
        actual_q_coin=estimates.q[actual_row],
    )


def collect_first_flops(
    env: Any,
    encoder: ObservationEncoder,
    policy: Policy,
    opponent_name: str,
    hands: int,
    seed: int,
) -> list[FirstFlopRecord]:
    """Play the actual deals once and record every first flop decision."""

    tracker = HistoryTracker()
    records: list[FirstFlopRecord] = []
    for hand in range(int(hands)):
        deal_seed = int(seed) + hand
        env.seed(deal_seed)
        opponent = make_opponent(opponent_name, seed=deal_seed)
        state, player = reset_hand(env)
        small_blind = small_blind_player(state)
        tracker.start_hand(state)
        notify_hand_start(opponent, state)

        key: bytes | None = None
        coin: int | None = getattr(opponent, "coin", None)

        while not env.is_over():
            if _is_first_flop_decision(state, player, recorded=key is not None):
                encoded = encoder.encode(
                    state, is_small_blind=(small_blind == player)
                )
                key = _betting_key(encoded.features)
            if player == 0:
                action = _action(policy, state, tracker, small_blind)
            else:
                action = _opponent_action(opponent, state, player)
            tracker.record(state, player, env.actions[action])
            state, player = env.step(action)

        if key is not None and coin is not None:
            records.append(
                FirstFlopRecord(
                    hand=hand,
                    deal_seed=deal_seed,
                    flat_key=key,
                    coin=int(coin),
                    payoff=float(env.get_payoffs()[0]),
                )
            )
    return records


def mixed_keys(records: Iterable[FirstFlopRecord]) -> set[bytes]:
    """Keys whose card-free summary was seen with both coins."""

    seen: dict[bytes, set[int]] = defaultdict(set)
    for record in records:
        seen[record.flat_key].add(record.coin)
    return {key for key, coins in seen.items() if len(coins) > 1}


def evaluate_records(
    env: Any,
    encoder: ObservationEncoder,
    policy: Policy,
    opponent_name: str,
    records: Sequence[FirstFlopRecord],
    wanted: set[bytes],
    runouts: int,
    seed: int,
) -> list[ForkResult]:
    """Replay each wanted deal to its fork and value the first flop action."""

    results: list[ForkResult] = []
    for record in records:
        if record.flat_key not in wanted:
            continue

        opponent = make_opponent(opponent_name, seed=record.deal_seed)
        tracker = HistoryTracker()
        fork = play_to_first_flop(
            env, policy, opponent, tracker, record.deal_seed
        )
        if fork is None:
            raise RuntimeError(
                f"deal {record.deal_seed} did not reach the recorded fork"
            )
        state, player, small_blind, steps = fork
        if player != 0:
            raise RuntimeError("first flop decision is not the agent's")
        encoded = encoder.encode(state, is_small_blind=(small_blind == 0))
        if _betting_key(encoded.features) != record.flat_key:
            raise RuntimeError("replayed fork does not match the recorded state")
        coin = getattr(opponent, "coin", None)
        if coin is None or int(coin) != record.coin:
            raise RuntimeError("replayed opponent coin does not match")

        snapshot = make_snapshot(
            env, state, tracker,
            hand=record.hand, deal_seed=record.deal_seed, coin=record.coin,
            steps=steps, small_blind=small_blind,
        )
        rng = np.random.default_rng(record.deal_seed + 1_048_573)
        pairs = sample_runouts(snapshot, runouts, rng)
        result = evaluate_fork(env, policy, opponent, snapshot, state, pairs)

        natural_actual = result.q_natural[
            snapshot.legal_actions.index(result.actual_action)
        ]
        if not np.isclose(natural_actual, record.payoff, atol=1e-9):
            raise RuntimeError(
                f"natural replay of deal {record.deal_seed} gave "
                f"{natural_actual!r}, actual hand gave {record.payoff!r}"
            )
        results.append(result)
    return results


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean, stderr, 95 % half-width and median of a per-hand sample."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    stderr = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
    half = Z_95 * stderr if np.isfinite(stderr) else float("nan")
    return mean, stderr, half, float(np.median(array))


def _report_seed(
    seed: int,
    checkpoint: str,
    records: Sequence[FirstFlopRecord],
    results: Sequence[ForkResult],
    hands: int,
    runouts: int,
) -> list[str]:
    lines: list[str] = []
    reached = len(records)
    conditional = [result.cross_fitted_ceiling for result in results]
    in_sample = [result.in_sample_ceiling for result in results]
    per_runout = [result.per_runout_ceiling for result in results]
    already_best = float(
        np.mean([result.best_action_in_sample == result.actual_action for result in results])
    ) if results else float("nan")

    lines.append(f"seed {seed}   checkpoint {checkpoint}")
    lines.append(
        f"  first flop decisions     : {reached} of {hands} deals "
        f"({reached / hands:.1%})"
    )
    lines.append(
        f"  card-free mixed (used)   : {len(results)} "
        f"({len(results) / reached:.1%} of first flops, "
        f"{len(results) / hands:.1%} of deals)"
    )
    conditional_mean, conditional_stderr, conditional_half, conditional_median = _summary(conditional)
    in_mean, in_stderr, in_half, _ = _summary(in_sample)
    runout_mean, runout_stderr, runout_half, _ = _summary(per_runout)
    per_deal = conditional_mean * len(results) / hands
    lines.append(
        f"  cross-fitted ceiling     : {conditional_mean:+.4f} "
        f"+/- {conditional_half:.4f} bb per mixed decision "
        f"(median {conditional_median:+.4f}, n={len(results)})"
    )
    lines.append(
        f"  in-sample ceiling        : {in_mean:+.4f} "
        f"+/- {in_half:.4f} bb per mixed decision (upper bound)"
    )
    lines.append(
        f"  per-runout max           : {runout_mean:+.4f} "
        f"+/- {runout_half:.4f} bb per mixed decision (runout luck included)"
    )
    lines.append(
        f"  actual action already best: {already_best:.1%} of mixed decisions"
    )
    better = [value for value in conditional if value > 0.0]
    if better:
        lines.append(
            f"  where a better action exists: {len(better)} of {len(results)} "
            f"mixed decisions, mean gain {float(np.mean(better)):+.4f} bb"
        )
    lines.append(
        f"  cross-fitted per deal    : {per_deal:+.4f} bb/hand over all deals"
    )
    return lines


def _report(
    args: argparse.Namespace,
    per_seed: Sequence[tuple[int, str, Sequence[FirstFlopRecord], Sequence[ForkResult]]],
) -> str:
    lines: list[str] = []
    lines.append("Poker AI Arena - value of perfect coin knowledge at the first flop")
    lines.append("")
    lines.append(f"opponent       : {args.opponent}")
    lines.append(f"arm            : {args.arm}   iteration {args.iteration}")
    lines.append(f"deals          : {args.hands} per checkpoint   base seed {args.seed}")
    lines.append(f"runouts/action : {args.runouts} sampled + 1 natural")
    lines.append("")

    for seed, checkpoint, records, results in per_seed:
        lines.extend(
            _report_seed(seed, checkpoint, records, results, args.hands, args.runouts)
        )
        lines.append("")

    all_cross = [
        result.cross_fitted_ceiling
        for _, _, _, results in per_seed
        for result in results
    ]
    all_in = [
        result.in_sample_ceiling
        for _, _, _, results in per_seed
        for result in results
    ]
    pooled_mean, pooled_stderr, pooled_half, _ = _summary(all_cross)
    pooled_in, pooled_in_stderr, pooled_in_half, _ = _summary(all_in)
    lines.append("pooled over checkpoints")
    lines.append(
        f"  cross-fitted ceiling     : {pooled_mean:+.4f} +/- {pooled_half:.4f} "
        f"bb per mixed decision (n={len(all_cross)})"
    )
    lines.append(
        f"  in-sample ceiling        : {pooled_in:+.4f} +/- {pooled_in_half:.4f} "
        f"bb per mixed decision (upper bound)"
    )
    lines.append("")
    lines.append("Read the cross-fitted row as a lower bound and the in-sample row as")
    lines.append("an upper bound.  Both compare the best first flop action with the")
    lines.append("action the deployed policy actually chose, on the same runouts.  A")
    lines.append("ceiling near the measured history ablation means even perfect")
    lines.append("coin knowledge at this decision could not have bought much; a")
    lines.append("ceiling far above it means the policy is leaving value on the")
    lines.append("table and the limitation is optimisation, not information.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.hands < 2:
        raise SystemExit("--hands must be at least 2")
    if args.runouts < 2:
        raise SystemExit("--runouts must be at least 2")
    if args.arm == "history":
        prefix = args.history_prefix
    else:
        prefix = args.control_prefix
    use_history = args.arm == "history"

    checkpoint_dir = Path(args.checkpoint_dir)
    env_play = make_environment(seed=args.seed)
    encoder = ObservationEncoder.from_environment(env_play)
    env_fork = make_environment(seed=args.seed, allow_step_back=True)

    per_seed = []
    for seed in args.seeds:
        name = f"{prefix}_seed{seed}_{args.iteration:04d}.pt"
        path = checkpoint_dir / name
        if not path.is_file():
            print(f"seed {seed}: missing checkpoint {name}")
            continue
        net, _ = load_checkpoint_b(path)
        net.eval()
        policy = AgentBPolicy(net, encoder, use_history=use_history)

        records = collect_first_flops(
            env_play, encoder, policy, args.opponent, args.hands, args.seed
        )
        wanted = mixed_keys(records)
        results = evaluate_records(
            env_fork, encoder, policy, args.opponent, records, wanted,
            args.runouts, args.seed,
        )
        per_seed.append((seed, name, records, results))

    if not per_seed:
        raise SystemExit("no checkpoints found")

    report = _report(args, per_seed)
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
