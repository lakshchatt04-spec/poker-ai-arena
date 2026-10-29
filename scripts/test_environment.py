"""Stage 1 smoke test for the RLCard limit Hold'em environment.

Run from the project root:

    python scripts\\test_environment.py

The script is intentionally an executable report: it prints what the simulator
actually exposes and asserts the facts the later agents will rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import rlcard

from poker_arena.environment import (
    ACTION_NAMES,
    CARD_SLOT_COUNT,
    ENVIRONMENT_ID,
    OBSERVATION_SIZE,
    action_mask,
    decode_observation,
    make_environment,
    play_random_hand,
    verify_observation_is_legal,
)

DEMO_SEED = 25  # this seed reaches the river with the random rollout below
N_CONSECUTIVE_HANDS = 100


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("Poker AI Arena - Stage 1 environment smoke test")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Interpreter : {sys.executable}")
    print(f"RLCard      : {rlcard.__version__}")
    print(f"Project     : {PROJECT_ROOT}")

    # ------------------------------------------------------------------
    _header("[1] Environment contract")
    env = make_environment(seed=DEMO_SEED)
    print(f"environment id   : {ENVIRONMENT_ID}")
    print(f"players          : {env.num_players}")
    print(f"action count     : {env.num_actions}")
    print(f"action names     : {env.actions}")
    print(
        "action ids       : "
        + ", ".join(f"{i}={name}" for i, name in enumerate(env.actions))
    )
    print(f"state shape      : {env.state_shape}")
    print(f"action shape     : {env.action_shape}")
    print(f"allow step back  : {env.allow_step_back}")

    assert env.num_players == 2
    assert tuple(env.actions) == ACTION_NAMES
    assert [int(np.prod(shape)) for shape in env.state_shape] == [
        OBSERVATION_SIZE,
        OBSERVATION_SIZE,
    ]

    # ------------------------------------------------------------------
    _header("[2] Initial state (fresh hand, before any action)")
    state, player_id = env.reset()
    raw_obs = state["raw_obs"]
    view = decode_observation(env, state["obs"])

    print(f"acting player        : P{player_id}")
    print(f"is_over              : {env.is_over()}")
    print(f"own hole cards       : {' '.join(raw_obs['hand'])}")
    print(f"public cards         : {' '.join(raw_obs['public_cards']) or '-'}")
    print(f"all chips            : {raw_obs['all_chips']}")
    print(f"my chips             : {raw_obs['my_chips']}")
    print(f"pot                  : {sum(raw_obs['all_chips'])}")
    print(f"cards encoded in obs : {' '.join(view.cards)}")
    print(f"raises per round     : {view.raises_per_round}")
    print(f"legal action ids     : {list(state['legal_actions'])}")
    print(f"legal action names   : {state['raw_legal_actions']}")
    print(f"action mask          : {action_mask(env, state).tolist()}")
    print(f"action record        : {state['action_record']}  (empty before action)")
    print(f"obs shape / dtype    : {state['obs'].shape} / {state['obs'].dtype}")
    print(f"obs unique values    : {np.unique(state['obs'])}")
    print(f"raw_obs keys         : {list(raw_obs)}")

    assert not env.is_over()
    assert state["action_record"] == []
    assert raw_obs["public_cards"] == []
    assert len(raw_obs["hand"]) == 2
    assert set(view.cards) == set(raw_obs["hand"])

    # ------------------------------------------------------------------
    _header("[3] One complete hand with random agents (gameplay states)")
    print("Each row is the state *before* that action is applied.")
    print("The initial state above has no history; every row below grows it by one.")
    hand_env = make_environment(seed=DEMO_SEED)
    result = play_random_hand(hand_env, np.random.default_rng(DEMO_SEED))

    for number, step in enumerate(result.steps, start=1):
        legal = ", ".join(
            f"{action_id}:{name}"
            for action_id, name in zip(
                step.legal_action_ids, step.legal_action_names
            )
        )
        history = (
            " ".join(f"P{pid}:{name}" for pid, name in step.history) or "-"
        )
        print(
            f"  {number:>2}. P{step.player_id} {step.action_name}"
            f"({step.action_id}) | pot={step.pot} | "
            f"board={' '.join(step.board) or '-'}"
        )
        print(f"      legal=[{legal}]")
        print(
            f"      obs cards={' '.join(step.observed_cards)} | "
            f"raises={step.raises_per_round} | history={history}"
        )

    print()
    print(f"final is_over     : {hand_env.is_over()}")
    print(f"decisions         : {len(result.steps)}")
    print(f"payoff type       : {type(result.payoffs).__name__}")
    print(
        f"final payoffs     : P0={result.payoffs[0]:+.1f}  "
        f"P1={result.payoffs[1]:+.1f}"
    )
    print(f"payoff sum        : {result.payoffs.sum():+.1f}")
    print(f"payoffs (raw)     : {result.payoffs!r}")

    assert hand_env.is_over()
    assert abs(float(result.payoffs.sum())) < 1e-9

    # ------------------------------------------------------------------
    _header(f"[4] {N_CONSECUTIVE_HANDS} consecutive hands on one environment")
    multi_env = make_environment(seed=2026)
    multi_rng = np.random.default_rng(2026)
    all_payoffs: list[np.ndarray] = []

    for hand_number in range(1, N_CONSECUTIVE_HANDS + 1):
        hand = play_random_hand(multi_env, multi_rng)
        if not multi_env.is_over():
            raise RuntimeError(f"hand {hand_number} did not terminate")
        if len(multi_env.action_recorder) != len(hand.steps):
            raise RuntimeError(
                f"hand {hand_number}: action recorder was not cleared by reset()"
            )
        all_payoffs.append(hand.payoffs)

    payoffs = np.asarray(all_payoffs)
    p0 = payoffs[:, 0]
    print(f"hands played           : {len(payoffs)}")
    print(f"payoff array shape     : {payoffs.shape}")
    print(f"zero-sum max error     : {np.abs(payoffs.sum(axis=1)).max():.2e}")
    print(f"P0 mean / sample std   : {p0.mean():+.3f} / {p0.std(ddof=1):.3f} bb")
    print(
        "P0 win / lose / tie    : "
        f"{(p0 > 0).mean():.1%} / {(p0 < 0).mean():.1%} / {(p0 == 0).mean():.1%}"
    )

    assert np.all(np.abs(payoffs.sum(axis=1)) < 1e-9)
    assert np.allclose(payoffs * 2, np.round(payoffs * 2))

    # ------------------------------------------------------------------
    _header("[5] Information hiding: only own hole cards + public cards")
    leak_env = make_environment(seed=DEMO_SEED)
    leak_rng = np.random.default_rng(DEMO_SEED)
    state, player_id = leak_env.reset()
    checked_decisions = 0

    print(
        "`env.get_perfect_information()` is used here for verification only."
    )
    print("It exposes both hands and must never be an agent input.\n")

    while not leak_env.is_over():
        for player in range(leak_env.num_players):
            verify_observation_is_legal(leak_env, player)

        if checked_decisions == 0:
            perfect = leak_env.get_perfect_information()
            public = set(perfect["public_card"] or [])
            print(f"public cards: {sorted(public) or '-'}")
            for player in range(leak_env.num_players):
                player_view = decode_observation(
                    leak_env, leak_env.get_state(player)["obs"]
                )
                opponent_private = (
                    set(perfect["hand_cards"][1 - player]) - public
                )
                overlap = set(player_view.cards) & opponent_private
                print(
                    f"  P{player} sees {list(player_view.cards)}; "
                    f"own={perfect['hand_cards'][player]}; "
                    f"opponent private={sorted(opponent_private)}; "
                    f"overlap={sorted(overlap) or 'none'}"
                )

            obs0 = leak_env.get_state(0)["obs"]
            obs1 = leak_env.get_state(1)["obs"]
            differing = np.flatnonzero(obs0 != obs1)
            print(
                f"  P0/P1 obs indices that differ: {differing.tolist()} "
                f"(all < {CARD_SLOT_COUNT}: "
                f"{bool((differing < CARD_SLOT_COUNT).all())})"
            )
            assert (differing < CARD_SLOT_COUNT).all()

        checked_decisions += 1
        legal = list(state["legal_actions"])
        state, player_id = leak_env.step(int(leak_rng.choice(legal)))

    print(
        f"\nchecked {checked_decisions} decision(s) for both players; "
        "no opponent hole card appeared in any observation"
    )

    # ------------------------------------------------------------------
    _header("[6] Verified API summary")
    print("reset()              -> (state dict, acting player id)")
    print("step(action_id)      -> (next state dict, next player id)")
    print("run(is_training)     -> (trajectories, payoffs)")
    print("state keys           -> legal_actions, obs, raw_obs,")
    print("                        raw_legal_actions, action_record")
    print("obs                  -> np.ndarray (72,), float64, one-hot")
    print("obs[0:52]            -> own hole cards + public cards")
    print("obs[52:72]           -> 4 betting rounds x 5 raise-count slots")
    print("legal_actions        -> OrderedDict(action id -> None)")
    print("raw_legal_actions    -> action names in the same order")
    print("payoffs              -> np.ndarray, chips / big blind, zero-sum")
    print("action_record        -> live list of (player_id, action_name)")
    print("timestep             -> cumulative across hands, not reset")

    print("\nAll Stage 1 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
