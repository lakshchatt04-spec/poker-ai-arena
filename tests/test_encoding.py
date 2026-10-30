"""Tests for Agent A's observation encoder."""

from __future__ import annotations

import copy

import numpy as np
import pytest

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
    EncodedState,
    ObservationEncoder,
    small_blind_player,
    validate_environment_contract,
)
from poker_arena.environment import action_mask, make_environment
from poker_arena.history import HistoryTracker, street_from_board_size


@pytest.fixture()
def env():
    return make_environment(seed=42)


@pytest.fixture()
def encoder(env):
    return ObservationEncoder.from_environment(env)


def _names(env) -> dict[int, str]:
    """Slot number -> RLCard card string."""

    return {int(index): card for card, index in env.card2index.items()}


def _state(
    env,
    *,
    hand: list[int],
    board: list[int],
    chips: list[int],
    my_chips: int,
    raises: tuple[int, int, int, int] = (0, 0, 0, 0),
    legal: tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    """Build a minimal state dict from slot numbers, with card strings inside."""

    names = _names(env)
    return {
        "raw_obs": {
            "hand": [names[index] for index in hand],
            "public_cards": [names[index] for index in board],
            "all_chips": list(chips),
            "my_chips": int(my_chips),
            "raise_nums": list(raises),
            "legal_actions": [],
        },
        "legal_actions": {int(action): None for action in legal},
    }


def _preferred_action(legal: list[int]) -> int:
    """Raise when possible, otherwise call, otherwise check."""

    for action in (1, 0, 3):
        if action in legal:
            return action
    raise AssertionError(f"no preferred action among {legal}")


# -- layout ---------------------------------------------------------------
def test_feature_vector_has_the_documented_shape_and_range(env, encoder):
    state, _ = env.reset()
    encoded = encoder.encode(state, is_small_blind=True)

    assert isinstance(encoded, EncodedState)
    assert encoded.features.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == 131
    assert encoded.features.dtype == np.float32
    assert np.all(np.isfinite(encoded.features))
    assert encoded.features.min() >= 0.0
    assert encoded.features.max() <= 1.0
    assert encoded.legal_mask.shape == (4,)
    assert encoded.legal_mask.dtype == bool


def test_hole_and_board_blocks_are_separate(env, encoder):
    """Two hands with the same five cards must not encode identically."""

    first = encoder.encode(
        _state(env, hand=[0, 1], board=[2, 3, 4], chips=[1, 2], my_chips=1),
        is_small_blind=True,
    )
    second = encoder.encode(
        _state(env, hand=[2, 3], board=[0, 1, 4], chips=[1, 2], my_chips=1),
        is_small_blind=True,
    )

    # RLCard's own obs[0:52] would be identical for both, because it merges
    # public cards and hole cards into one multi-hot.
    assert set(np.flatnonzero(first.features[HOLE_SLICE])) == {0, 1}
    assert set(np.flatnonzero(first.features[BOARD_SLICE])) == {2, 3, 4}
    assert set(np.flatnonzero(second.features[HOLE_SLICE])) == {2, 3}
    assert set(np.flatnonzero(second.features[BOARD_SLICE])) == {0, 1, 4}
    assert not np.array_equal(first.features, second.features)


def test_hole_block_marks_only_the_own_cards(env, encoder):
    env.reset()
    perfect = env.get_perfect_information()

    for player in range(env.num_players):
        view = env.get_state(player)
        encoded = encoder.encode(view, is_small_blind=False)
        hot = set(np.flatnonzero(encoded.features[HOLE_SLICE]))

        own = {env.card2index[card] for card in perfect["hand_cards"][player]}
        assert hot == own
        assert len(hot) == 2


def test_board_block_tracks_the_public_cards(env, encoder):
    state, player_id = env.reset()
    rng = np.random.default_rng(0)
    seen_streets = set()

    while not env.is_over():
        raw_obs = state["raw_obs"]
        encoded = encoder.encode(state, is_small_blind=False)
        hot = set(np.flatnonzero(encoded.features[BOARD_SLICE]))
        public = {env.card2index[card] for card in raw_obs["public_cards"]}

        assert hot == public
        assert int(encoded.features[BOARD_SLICE].sum()) == len(raw_obs["public_cards"])
        seen_streets.add(encoded.street)

        legal = list(state["legal_actions"])
        state, player_id = env.step(int(rng.choice(legal)))

    assert seen_streets


def test_raise_block_is_one_hot_on_every_street(env, encoder):
    encoded = encoder.encode(
        _state(
            env,
            hand=[0, 1],
            board=[2, 3, 4],
            chips=[1, 2],
            my_chips=1,
            raises=(2, 4, 0, 1),
        ),
        is_small_blind=False,
    )

    block = encoded.features[RAISE_SLICE].reshape(4, 5)
    assert np.all(block.sum(axis=1) == 1)
    assert [int(np.argmax(row)) for row in block] == [2, 4, 0, 1]


def test_street_one_hot_matches_the_board_size(env, encoder):
    for board_size, expected in ((0, 0), (3, 1), (4, 2), (5, 3)):
        encoded = encoder.encode(
            _state(
                env,
                hand=[0, 1],
                board=list(range(10, 10 + board_size)),
                chips=[1, 2],
                my_chips=1,
            ),
            is_small_blind=False,
        )
        assert encoded.street == expected
        assert int(np.argmax(encoded.features[STREET_SLICE])) == expected
        assert encoded.features[STREET_SLICE].sum() == 1.0


def test_impossible_board_size_is_rejected(env, encoder):
    for board_size in (1, 2, 6):
        with pytest.raises(ValueError):
            encoder.encode(
                _state(
                    env,
                    hand=[0, 1],
                    board=list(range(10, 10 + board_size)),
                    chips=[1, 2],
                    my_chips=1,
                ),
                is_small_blind=False,
            )
    with pytest.raises(ValueError):
        street_from_board_size(2)


# -- numeric features -----------------------------------------------------
def test_pot_and_to_call_use_big_blind_units(env, encoder):
    encoded = encoder.encode(
        _state(env, hand=[0, 1], board=[], chips=[1, 2], my_chips=1),
        is_small_blind=True,
    )

    # 3 chips = 1.5 bb in the pot, 1 chip to call = 0.5 bb.
    assert encoded.features[POT_INDEX] == pytest.approx(1.5 / POT_CEILING_BB)
    assert encoded.features[TO_CALL_INDEX] == pytest.approx(0.5 / CALL_CEILING_BB)

    # Anything above the ceiling clips instead of leaving the input range.
    deep = encoder.encode(
        _state(env, hand=[0, 1], board=[], chips=[10, 4], my_chips=4),
        is_small_blind=False,
    )
    assert deep.features[POT_INDEX] == pytest.approx(7.0 / POT_CEILING_BB)
    assert deep.features[TO_CALL_INDEX] == pytest.approx(1.0)


def test_to_call_is_zero_when_the_chips_are_level(env, encoder):
    state, player_id = env.reset()
    state, player_id = env.step(0)  # the small blind calls

    encoded = encoder.encode(state, is_small_blind=False)
    assert state["raw_obs"]["all_chips"] == [2, 2]
    assert encoded.features[TO_CALL_INDEX] == pytest.approx(0.0)


def test_pot_feature_reaches_its_ceiling_under_maximal_raising(env, encoder):
    """The 48 bb ceiling is reachable, so the feature is not dead weight."""

    largest = 0.0
    for _ in range(100):
        state, player_id = env.reset()
        while not env.is_over():
            encoded = encoder.encode(state, is_small_blind=False)
            largest = max(largest, float(encoded.features[POT_INDEX]))
            legal = list(state["legal_actions"])
            state, player_id = env.step(_preferred_action(legal))

    assert largest == pytest.approx(1.0)


def test_pot_ceiling_matches_the_measured_maximum(env):
    """Guards the constants: a different betting config must fail this test."""

    largest_chips = 0
    for _ in range(100):
        state, player_id = env.reset()
        while not env.is_over():
            largest_chips = max(largest_chips, sum(state["raw_obs"]["all_chips"]))
            legal = list(state["legal_actions"])
            state, player_id = env.step(_preferred_action(legal))

    assert largest_chips / 2 == pytest.approx(POT_CEILING_BB)


# -- position -------------------------------------------------------------
def test_small_blind_player_agrees_with_blinds_and_history_tracker(env, encoder):
    tracker = HistoryTracker()

    for _ in range(50):
        state, player_id = env.reset()
        small_blind = small_blind_player(state)

        assert state["raw_obs"]["all_chips"].index(1) == small_blind
        assert state["raw_obs"]["all_chips"].count(2) == 1
        assert player_id == small_blind

        tracker.start_hand(state)
        token = tracker.build_token(state, small_blind, "call")
        other = tracker.build_token(state, 1 - small_blind, "call")
        assert token.is_small_blind is True
        assert other.is_small_blind is False

        for flag in (True, False):
            encoded = encoder.encode(state, is_small_blind=flag)
            assert encoded.features[SMALL_BLIND_INDEX] == (1.0 if flag else 0.0)

        while not env.is_over():
            legal = list(state["legal_actions"])
            state, player_id = env.step(int(legal[0]))


def test_small_blind_player_rejects_a_state_from_mid_hand(env):
    state, player_id = env.reset()
    state, player_id = env.step(0)

    with pytest.raises(ValueError):
        small_blind_player(state)


# -- legality -------------------------------------------------------------
def test_legal_mask_matches_the_environment_on_every_decision(env, encoder):
    state, player_id = env.reset()
    decisions = 0

    while not env.is_over():
        encoded = encoder.encode(state, is_small_blind=False)
        expected = action_mask(env, state).astype(bool)

        assert np.array_equal(encoded.legal_mask, expected)
        assert sorted(np.flatnonzero(encoded.legal_mask).tolist()) == sorted(
            int(action) for action in state["legal_actions"]
        )
        assert encoded.legal_mask.any()

        decisions += 1
        legal = list(state["legal_actions"])
        state, player_id = env.step(int(legal[0]))

    assert decisions > 0


def test_masked_actions_get_no_share_of_the_probability(env, encoder):
    """The mask is usable directly: illegal logits can be driven to -infinity."""

    state, _ = env.reset()
    encoded = encoder.encode(state, is_small_blind=True)

    rng = np.random.default_rng(0)
    logits = rng.normal(size=4)
    masked = np.where(encoded.legal_mask, logits, -np.inf)
    probabilities = np.exp(masked - masked.max())
    probabilities /= probabilities.sum()

    assert np.all(probabilities[~encoded.legal_mask] == 0.0)
    assert probabilities.sum() == pytest.approx(1.0)


# -- robustness -----------------------------------------------------------
def test_encoding_does_not_mutate_the_state(env, encoder):
    state, _ = env.reset()
    before_raw = copy.deepcopy(state["raw_obs"])
    before_history = copy.deepcopy(state["action_record"])

    encoded = encoder.encode(state, is_small_blind=True)
    encoded.features[0] = 1.0

    assert state["raw_obs"] == before_raw
    assert state["action_record"] == before_history


def test_encoder_cannot_see_opponent_hole_cards(env, encoder):
    state, player_id = env.reset()

    while not env.is_over():
        perfect = env.get_perfect_information()
        public = {env.card2index[card] for card in perfect["public_card"] or []}

        for player in range(env.num_players):
            encoded = encoder.encode(env.get_state(player), is_small_blind=False)
            own = {env.card2index[card] for card in perfect["hand_cards"][player]}
            opponent_private = {
                env.card2index[card] for card in perfect["hand_cards"][1 - player]
            } - public

            encoded_cards = set(np.flatnonzero(encoded.features[HOLE_SLICE])) | set(
                np.flatnonzero(encoded.features[BOARD_SLICE])
            )
            assert encoded_cards == own | public
            assert not encoded_cards & opponent_private

        legal = list(state["legal_actions"])
        state, player_id = env.step(int(legal[0]))


def test_invalid_states_are_rejected(env, encoder):
    with pytest.raises(ValueError):
        encoder.encode(
            _state(env, hand=[0, 1], board=[], chips=[1, 2], my_chips=5),
            is_small_blind=False,
        )
    with pytest.raises(ValueError):
        encoder.encode(
            _state(
                env,
                hand=[0, 1],
                board=[],
                chips=[1, 2],
                my_chips=1,
                raises=(0, 5, 0, 0),
            ),
            is_small_blind=False,
        )

    overlapping = _state(env, hand=[0, 1], board=[1, 2, 3], chips=[1, 2], my_chips=1)
    with pytest.raises(ValueError):
        encoder.encode(overlapping, is_small_blind=False)

    unknown = _state(env, hand=[0, 1], board=[], chips=[1, 2], my_chips=1)
    unknown["raw_obs"]["hand"][0] = "XX"
    with pytest.raises(ValueError):
        encoder.encode(unknown, is_small_blind=False)


def test_encode_batch_aligns_features_and_masks(env, encoder):
    states = []
    flags = []
    state, player_id = env.reset()

    while not env.is_over():
        states.append(state)
        flags.append(True)
        legal = list(state["legal_actions"])
        state, player_id = env.step(int(legal[0]))

    features, masks = encoder.encode_batch(states, flags)
    assert features.shape == (len(states), FEATURE_DIM)
    assert masks.shape == (len(states), 4)
    assert features.dtype == np.float32
    assert np.array_equal(
        features[0], encoder.encode(states[0], is_small_blind=True).features
    )

    with pytest.raises(ValueError):
        encoder.encode_batch(states, flags[:-1])

    empty_features, empty_masks = encoder.encode_batch([], [])
    assert empty_features.shape == (0, FEATURE_DIM)
    assert empty_masks.shape == (0, 4)


def test_environment_contract_matches_the_encoder_ceilings(env):
    validate_environment_contract(env)
