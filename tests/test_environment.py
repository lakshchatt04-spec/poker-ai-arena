"""Stage 1 tests for the RLCard limit Hold'em environment contract."""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.environment import (
    ACTION_NAMES,
    CARD_SLOT_COUNT,
    NUM_BETTING_ROUNDS,
    OBSERVATION_SIZE,
    RAISE_BLOCK_SIZE,
    action_mask,
    decode_observation,
    make_environment,
    play_random_hand,
    verify_observation_is_legal,
)


@pytest.fixture()
def env():
    """A fresh two-player limit Hold'em environment per test."""

    return make_environment(seed=42)


def test_environment_initializes_with_two_players(env):
    assert env.num_players == 2
    assert env.num_actions == 4
    assert tuple(env.actions) == ACTION_NAMES
    assert [int(np.prod(shape)) for shape in env.state_shape] == [
        OBSERVATION_SIZE,
        OBSERVATION_SIZE,
    ]

    state, player_id = env.reset()
    assert player_id in (0, 1)
    assert env.get_player_id() == player_id
    assert state["obs"].shape == (OBSERVATION_SIZE,)
    assert len(state["raw_obs"]["all_chips"]) == env.num_players


def test_complete_hand_terminates_and_returns_zero_sum_payoffs(env):
    result = play_random_hand(env, np.random.default_rng(123))

    assert env.is_over()
    assert isinstance(result.payoffs, np.ndarray)
    assert result.payoffs.shape == (env.num_players,)
    assert np.all(np.isfinite(result.payoffs))
    assert result.payoffs.sum() == pytest.approx(0.0, abs=1e-9)

    # RLCard divides integer chip payoffs by the big blind (2 chips).
    assert np.allclose(result.payoffs * 2, np.round(result.payoffs * 2))
    assert result.steps


def test_multiple_hands_can_be_played_consecutively(env):
    rng = np.random.default_rng(7)
    all_payoffs = []

    for _ in range(25):
        result = play_random_hand(env, rng)

        assert env.is_over()
        assert len(env.action_recorder) == len(result.steps)
        assert env.action_recorder == [
            (step.player_id, step.action_name) for step in result.steps
        ]
        all_payoffs.append(result.payoffs)

    payoffs = np.asarray(all_payoffs)
    assert payoffs.shape == (25, env.num_players)
    assert np.allclose(payoffs.sum(axis=1), 0.0, atol=1e-9)


def test_legal_actions_are_exposed_as_parallel_id_and_name_lists(env):
    state, _ = env.reset()
    legal_actions = state["legal_actions"]

    assert hasattr(legal_actions, "keys")
    action_ids = list(legal_actions.keys())
    assert action_ids
    assert all(isinstance(action_id, int) for action_id in action_ids)
    assert all(0 <= action_id < env.num_actions for action_id in action_ids)
    assert all(value is None for value in legal_actions.values())

    action_names = state["raw_legal_actions"]
    assert len(action_names) == len(action_ids)
    assert action_names == [env.actions[action_id] for action_id in action_ids]
    assert set(action_names) <= set(env.actions)

    mask = action_mask(env, state)
    assert mask.shape == (env.num_actions,)
    assert sorted(np.flatnonzero(mask).tolist()) == sorted(action_ids)


def test_initial_state_is_distinct_from_gameplay_states(env):
    state, player_id = env.reset()

    assert not env.is_over()
    assert state["action_record"] == []
    assert state["raw_obs"]["public_cards"] == []
    assert len(state["raw_obs"]["hand"]) == 2

    result = play_random_hand(env, np.random.default_rng(1))

    # Each decision sees exactly the actions taken before it, never a future
    # action and never history from the previous hand.
    assert [len(step.history) for step in result.steps] == list(
        range(len(result.steps))
    )
    assert result.steps[0].history == ()


def test_observation_layout_is_one_hot(env):
    state, _ = env.reset()
    obs = state["obs"]

    assert obs.shape == (OBSERVATION_SIZE,)
    assert set(np.unique(obs)) <= {0.0, 1.0}

    raw_obs = state["raw_obs"]
    expected_cards = len(raw_obs["hand"]) + len(raw_obs["public_cards"])
    assert np.count_nonzero(obs[:CARD_SLOT_COUNT]) == expected_cards

    raise_blocks = obs[CARD_SLOT_COUNT:].reshape(
        NUM_BETTING_ROUNDS, RAISE_BLOCK_SIZE
    )
    assert np.all(raise_blocks.sum(axis=1) == 1)

    view = decode_observation(env, obs)
    assert set(view.cards) == set(raw_obs["hand"]) | set(raw_obs["public_cards"])


def test_observations_do_not_leak_opponent_hole_cards(env):
    state, _ = env.reset()
    rng = np.random.default_rng(25)

    while not env.is_over():
        for player_id in range(env.num_players):
            verify_observation_is_legal(env, player_id)

        obs0 = env.get_state(0)["obs"]
        obs1 = env.get_state(1)["obs"]
        differing = np.flatnonzero(obs0 != obs1)
        assert (differing < CARD_SLOT_COUNT).all()

        legal_actions = list(state["legal_actions"])
        state, _ = env.step(int(rng.choice(legal_actions)))
