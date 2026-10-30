"""Agent A observation-encoder smoke test.

Run from the project root:

    python scripts\\test_encoding.py

Prints the 131-number vector block by block for a real hand and asserts the
properties the policy network will rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from poker_arena.encoding import (
    BOARD_SLICE,
    CALL_CEILING_BB,
    FEATURE_DIM,
    HOLE_SLICE,
    POT_CEILING_BB,
    POT_INDEX,
    RAISE_SLICE,
    SMALL_BLIND_INDEX,
    STREET_SLICE,
    TO_CALL_INDEX,
    ObservationEncoder,
    small_blind_player,
    validate_environment_contract,
)
from poker_arena.environment import action_mask, make_environment

DEMO_SEED = 25
N_RANDOM_HANDS = 100


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _blocks(encoded) -> None:
    features = encoded.features
    print(f"  hole  [0:52]    cards={np.flatnonzero(features[HOLE_SLICE]).tolist()}")
    print(f"  board [52:104]  cards={np.flatnonzero(features[BOARD_SLICE]).tolist()}")
    block = features[RAISE_SLICE].reshape(4, 5).astype(int)
    print("  raise [104:124] per street:")
    for street, row in enumerate(block):
        print(f"                  street {street}: {row.tolist()}")
    print(f"  street [124:128] one-hot={features[STREET_SLICE].tolist()}")
    print(f"  pot    [128]     {features[POT_INDEX]:.4f}")
    print(f"  to_call[129]     {features[TO_CALL_INDEX]:.4f}")
    print(f"  blinds [130]     {features[SMALL_BLIND_INDEX]:.0f}")


def main() -> int:
    print("Poker AI Arena - Agent A observation encoder smoke test")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Project : {PROJECT_ROOT}")

    # ------------------------------------------------------------------
    _header("[1] Encoder contract")
    env = make_environment(seed=DEMO_SEED)
    validate_environment_contract(env)
    encoder = ObservationEncoder.from_environment(env)

    print(f"deck size          : {len(env.card2index)}")
    print(f"feature width      : {FEATURE_DIM}")
    print(f"pot ceiling        : {POT_CEILING_BB:g} bb")
    print(f"to_call ceiling    : {CALL_CEILING_BB:g} bb")
    print("layout             : 0:52 hole | 52:104 board | 104:124 raises |")
    print("                     124:128 street | 128 pot | 129 to_call |")
    print("                     130 small blind")
    assert FEATURE_DIM == 131

    # ------------------------------------------------------------------
    _header("[2] One hand, decision by decision")
    print("Each row is the state *before* that action is applied.")
    rng = np.random.default_rng(DEMO_SEED)
    state, player_id = env.reset()
    small_blind = small_blind_player(state)
    print(f"small blind this hand: P{small_blind} (randomised by RLCard)")
    print(f"first actor          : P{player_id}")

    decisions = 0
    while not env.is_over():
        encoded = encoder.encode(
            state, is_small_blind=(player_id == small_blind)
        )
        raw_obs = state["raw_obs"]
        mask = action_mask(env, state).astype(bool)

        if decisions < 6 or encoded.street >= 3:
            print()
            print(
                f"P{player_id} | chips={raw_obs['all_chips']} "
                f"| board={raw_obs['public_cards'] or '-'} "
                f"| raises={raw_obs['raise_nums']} "
                f"| street={encoded.street}"
            )
            _blocks(encoded)
            print(
                f"  mask {encoded.legal_mask.astype(int).tolist()} "
                f"== env action_mask {mask.astype(int).tolist()}"
            )

        assert np.array_equal(encoded.legal_mask, mask)
        assert encoded.features.min() >= 0.0 and encoded.features.max() <= 1.0

        decisions += 1
        legal = list(state["legal_actions"])
        state, player_id = env.step(int(rng.choice(legal)))

    print(f"\ndecisions in this hand: {decisions}")
    print(f"final payoffs         : {env.get_payoffs()}")

    # ------------------------------------------------------------------
    _header("[3] Why hole and board are separate blocks")
    card = {index: name for name, index in env.card2index.items()}
    hand_a = [card[0], card[1]]
    board_a = [card[2], card[3], card[4]]
    hand_b = [card[2], card[3]]
    board_b = [card[0], card[1], card[4]]

    print(f"hand A: hole={' '.join(hand_a)} board={' '.join(board_a)}")
    print(f"hand B: hole={' '.join(hand_b)} board={' '.join(board_b)}")
    print("both sets of five cards are identical, so RLCard's obs[0:52] matches")
    print("and the network could not tell them apart.\n")

    for label, hand, board in (("A", hand_a, board_a), ("B", hand_b, board_b)):
        view = encoder.encode(
            {
                "raw_obs": {
                    "hand": hand,
                    "public_cards": board,
                    "all_chips": [1, 2],
                    "my_chips": 1,
                    "raise_nums": [0, 0, 0, 0],
                    "legal_actions": [],
                },
                "legal_actions": {0: None, 1: None, 2: None},
            },
            is_small_blind=True,
        )
        print(
            f"  encoded {label}: hole="
            f"{np.flatnonzero(view.features[HOLE_SLICE]).tolist()} "
            f"board={np.flatnonzero(view.features[BOARD_SLICE]).tolist()}"
        )

    # ------------------------------------------------------------------
    _header(f"[4] {N_RANDOM_HANDS} random hands")
    check_env = make_environment(seed=2026)
    check_encoder = ObservationEncoder.from_environment(check_env)
    check_rng = np.random.default_rng(2026)

    min_value, max_value = 1.0, 0.0
    street_visits = [0, 0, 0, 0]
    mask_mismatches = 0
    max_pot = 0.0

    for _ in range(N_RANDOM_HANDS):
        state, player_id = check_env.reset()
        small_blind = small_blind_player(state)
        while not check_env.is_over():
            encoded = check_encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            min_value = min(min_value, float(encoded.features.min()))
            max_value = max(max_value, float(encoded.features.max()))
            street_visits[encoded.street] += 1
            max_pot = max(max_pot, float(encoded.features[POT_INDEX]))
            if not np.array_equal(
                encoded.legal_mask, action_mask(check_env, state).astype(bool)
            ):
                mask_mismatches += 1

            legal = list(state["legal_actions"])
            state, player_id = check_env.step(int(check_rng.choice(legal)))

    print(f"hands                : {N_RANDOM_HANDS}")
    print(f"value range          : [{min_value:.3f}, {max_value:.3f}]")
    print(f"decisions by street  : {street_visits}")
    print(f"largest pot feature  : {max_pot:.4f} (ceiling 1.0)")
    print(f"mask mismatches      : {mask_mismatches}")

    assert min_value >= 0.0 and max_value <= 1.0
    assert mask_mismatches == 0
    assert all(count > 0 for count in street_visits)
    assert max_pot > 0.5

    # ------------------------------------------------------------------
    _header("[5] API summary")
    print("encoder = ObservationEncoder.from_environment(env)")
    print("small  = small_blind_player(reset_state)      # once per hand")
    print("out    = encoder.encode(state, is_small_blind=(player_id == small))")
    print("out    -> EncodedState(features float32[131], legal_mask bool[4], street)")
    print("batch  = encoder.encode_batch(states, flags)  -> (n,131), (n,4)")
    print("mask   -> applied to logits before softmax; never fed in as a feature")

    print("\nAll observation-encoder checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
