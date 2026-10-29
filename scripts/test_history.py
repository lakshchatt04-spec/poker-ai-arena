"""Smoke test for the Agent B history encoding.

Run from the project root:

    python scripts\\test_history.py

The script is an executable report: it plays real hands, prints the token table
one hand produced, and asserts that the encoding agrees with RLCard.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from poker_arena.environment import ACTION_NAMES, make_environment
from poker_arena.history import (
    ACTION_SLICE,
    ACTOR_SLICE,
    IN_POSITION_INDEX,
    IS_SELF_INDEX,
    MAX_HISTORY_TOKENS,
    POT_CEILING_BB,
    POT_INDEX,
    RAISES_INDEX,
    RAISE_LIMIT,
    SMALL_BLIND_INDEX,
    STREET_NAMES,
    STREET_SLICE,
    TOKEN_DIM,
    TO_CALL_INDEX,
    HistoryTracker,
)

DEMO_SEED = 25
N_HANDS = 200


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("Poker AI Arena - Agent B history encoding smoke test")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Interpreter     : {sys.executable}")
    print(f"Project         : {PROJECT_ROOT}")
    print(f"token dimension : {TOKEN_DIM}")
    print(f"max tokens/hand : {MAX_HISTORY_TOKENS}")

    # ------------------------------------------------------------------
    _header("[1] Token layout")
    print(f"  {'idx':<5}{'width':<7}feature")
    print(f"  {ACTOR_SLICE.start}-{ACTOR_SLICE.stop - 1:<4}{2:<7}actor one-hot (P0, P1)")
    print(f"  {IS_SELF_INDEX:<5}{1:<7}is_self (actor is the player deciding now)")
    print(f"  {ACTION_SLICE.start}-{ACTION_SLICE.stop - 1:<4}{4:<7}action one-hot {ACTION_NAMES}")
    print(f"  {STREET_SLICE.start}-{STREET_SLICE.stop - 1:<4}{4:<7}street one-hot {STREET_NAMES}")
    print(f"  {RAISES_INDEX:<5}{1:<7}raises on this street / {RAISE_LIMIT}")
    print(f"  {POT_INDEX:<5}{1:<7}pot in big blinds / {POT_CEILING_BB:.0f}")
    print(f"  {TO_CALL_INDEX:<5}{1:<7}to-call in big blinds / 2")
    print(f"  {SMALL_BLIND_INDEX:<5}{1:<7}actor posted the small blind")
    print(f"  {IN_POSITION_INDEX:<5}{1:<7}actor is not first to act on this street")

    # ------------------------------------------------------------------
    _header(f"[2] One tracked hand (seed {DEMO_SEED})")
    env = make_environment(seed=DEMO_SEED)
    rng = np.random.default_rng(DEMO_SEED)
    tracker = HistoryTracker()
    state, player_id = env.reset()
    tracker.start_hand(state)

    print(f"small blind      : P{tracker.sb_player}")
    print(f"big blind        : {tracker.big_blind} chips")
    print()
    print(
        f"  {'#':>2}  {'viewer':<8}{'seen':>5}  {'action':<8}{'street':<9}"
        f"{'raises':>7}{'pot_bb':>8}{'call_bb':>9}{'SB':>5}{'IP':>5}"
    )

    decisions = 0
    while not env.is_over():
        decisions += 1
        sequence = tracker.tokens(viewer=player_id)
        assert sequence.length == int(sequence.mask.sum())

        legal_ids = list(state["legal_actions"])
        action_id = int(rng.choice(legal_ids))
        action_name = env.actions[action_id]
        token = tracker.record(state, player_id, action_name)

        print(
            f"  {decisions:>2}  {'P' + str(player_id):<8}{sequence.length:>5}  "
            f"{action_name:<8}{token.street_name:<9}{token.raises_this_street:>7}"
            f"{token.pot_bb:>8.1f}{token.to_call_bb:>9.1f}"
            f"{int(token.is_small_blind):>5}{int(token.is_in_position):>5}"
        )
        state, player_id = env.step(action_id)

    payoffs = np.asarray(env.get_payoffs(), dtype=float)
    print()
    openers: dict[int, int] = {}
    for token in tracker.actions:
        openers.setdefault(token.street, token.actor)
    print(
        "street openers   : "
        + ", ".join(
            f"{STREET_NAMES[street]}=P{actor}"
            for street, actor in sorted(openers.items())
        )
    )
    print(
        "note             : each street opens with the player after the one "
        "who closed the previous street,"
    )
    print("                   so in-position depends on the action order, not the player id")
    print(f"decisions        : {decisions}")
    print(f"tokens stored    : {tracker.length} (dropped {tracker.dropped})")
    print(f"payoffs          : P0={payoffs[0]:+.1f}  P1={payoffs[1]:+.1f} bb")

    # ------------------------------------------------------------------
    _header("[3] Verification against RLCard")
    recorded = [(int(pid), str(name)) for pid, name in env.action_recorder]
    replayed = [(token.actor, token.action) for token in tracker.actions]
    print(f"action_record    : {len(recorded)} actions")
    print(f"token stream     : {len(replayed)} tokens")
    print(f"identical order  : {replayed == recorded}")
    assert replayed == recorded

    sequence = tracker.tokens(viewer=0)
    print(f"tensor shape     : {sequence.tokens.shape}")
    print(f"mask shape       : {sequence.mask.shape}")
    print(f"real rows        : {sequence.real_tokens().shape}")
    print(f"mask is a prefix : {bool(sequence.mask[: sequence.length].all())}")
    assert sequence.mask[: sequence.length].all()
    assert not sequence.mask[sequence.length :].any()

    if tracker.actions:
        first = tracker.actions[0]
        print()
        print(
            f"first token      : P{first.actor} {first.action} on "
            f"{first.street_name} (pot {first.pot_bb:.1f} bb, "
            f"to call {first.to_call_bb:.1f} bb)"
        )
        print(f"encoded for P0   : {np.array2string(sequence.tokens[0], precision=2)}")
        print(f"encoded for P1   : {np.array2string(tracker.tokens(viewer=1).tokens[0], precision=2)}")
        print(
            "differ only at is_self: "
            f"{np.flatnonzero(sequence.tokens[0] != tracker.tokens(viewer=1).tokens[0]).tolist()}"
        )
        assert np.flatnonzero(
            sequence.tokens[0] != tracker.tokens(viewer=1).tokens[0]
        ).tolist() == [IS_SELF_INDEX]

    # ------------------------------------------------------------------
    _header(f"[4] {N_HANDS} random hands")
    stats_env = make_environment(seed=2026)
    stats_rng = np.random.default_rng(2026)

    lengths: list[int] = []
    empty_decisions = 0
    total_decisions = 0
    dropped = 0
    max_pot = 0.0
    max_call = 0.0
    small_blind_counts = [0, 0]
    in_position = 0
    is_self = 0

    for _ in range(N_HANDS):
        hand_tracker = HistoryTracker()
        state, player_id = stats_env.reset()
        hand_tracker.start_hand(state)
        small_blind_counts[hand_tracker.sb_player] += 1

        while not stats_env.is_over():
            seq = hand_tracker.tokens(viewer=player_id)
            if seq.length == 0:
                empty_decisions += 1
            total_decisions += 1

            legal_ids = [int(action) for action in state["legal_actions"]]
            action_id = int(stats_rng.choice(legal_ids))
            hand_tracker.record(state, player_id, stats_env.actions[action_id])
            state, player_id = stats_env.step(action_id)

        lengths.append(hand_tracker.length)
        dropped += hand_tracker.dropped

        real = hand_tracker.tokens(viewer=0).real_tokens()
        assert real.shape[0] == hand_tracker.length
        assert np.all(real >= 0.0) and np.all(real <= 1.0)
        max_pot = max(max_pot, float(real[:, POT_INDEX].max(initial=0.0)))
        max_call = max(max_call, float(real[:, TO_CALL_INDEX].max(initial=0.0)))
        in_position += int(real[:, IN_POSITION_INDEX].sum())
        is_self += int(real[:, IS_SELF_INDEX].sum())

        assert [(token.actor, token.action) for token in hand_tracker.actions] == [
            (int(pid), str(name)) for pid, name in stats_env.action_recorder
        ]

    length_array = np.asarray(lengths)
    print(f"hands                   : {len(lengths)}")
    print(f"decisions               : {total_decisions}")
    print(
        f"tokens per hand         : min {length_array.min()}  "
        f"mean {length_array.mean():.2f}  max {length_array.max()}"
    )
    print(f"capacity                : {MAX_HISTORY_TOKENS}  (dropped {dropped})")
    print(f"decisions with no history: {empty_decisions}")
    print(
        f"small blind hands       : P0 {small_blind_counts[0]}  "
        f"P1 {small_blind_counts[1]}"
    )
    print(f"in-position tokens      : {in_position} / {length_array.sum()}")
    print(f"is_self tokens (P0 view): {is_self}")
    print(f"max pot / call feature  : {max_pot:.3f} / {max_call:.3f} of 1.0")

    assert length_array.max() <= MAX_HISTORY_TOKENS
    assert dropped == 0
    assert set(small_blind_counts) and all(count > 0 for count in small_blind_counts)
    assert max_pot <= 1.0 and max_call <= 1.0
    assert 0 < empty_decisions < total_decisions

    # ------------------------------------------------------------------
    _header("[5] Summary")
    print(f"token              : {TOKEN_DIM} floats")
    print(f"sequence           : ({MAX_HISTORY_TOKENS}, {TOKEN_DIM}) + mask + length")
    print("empty history      : all-False mask, GRU starts from zero state")
    print("overflow           : keep the newest 32 tokens")
    print("recorded per action: actor, action, street, raises, pot, to-call, SB, IP")

    print("\nAll history-encoding checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
