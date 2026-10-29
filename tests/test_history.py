"""Tests for the Agent B history encoding."""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.environment import ACTION_NAMES, make_environment
from poker_arena.history import (
    ACTION_SLICE,
    ACTOR_SLICE,
    CALL_CEILING_BB,
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
    ActionToken,
    HistoryTracker,
    encode_token,
    street_from_board_size,
)


@pytest.fixture()
def env():
    """A fresh two-player limit Hold'em environment per test."""

    return make_environment(seed=42)


def sample_token(**overrides) -> ActionToken:
    values = dict(
        actor=0,
        action="raise",
        street=2,
        raises_this_street=1,
        pot_bb=6.5,
        to_call_bb=1.0,
        is_small_blind=True,
        is_in_position=False,
    )
    values.update(overrides)
    return ActionToken(**values)


def play_tracked_hand(env, rng):
    """Play one hand, snapshotting the tracker at every decision.

    Returns the finished tracker plus one record per decision, each holding the
    sequence the agent would have seen, the matching ``action_record`` length
    and the raw observation *before* the action was applied.
    """

    tracker = HistoryTracker()
    state, player_id = env.reset()
    tracker.start_hand(state)

    decisions = []
    while not env.is_over():
        legal_ids = [int(action) for action in state["legal_actions"]]
        decisions.append(
            {
                "player": player_id,
                "sequence": tracker.tokens(viewer=player_id),
                "record_length": len(state["action_record"]),
                "raw_obs": state["raw_obs"],
                # RLCard hands out its live `history_raise_nums` list, so copy it.
                "raises": list(state["raw_obs"]["raise_nums"]),
            }
        )
        action_id = int(rng.choice(legal_ids))
        tracker.record(state, player_id, env.actions[action_id])
        state, player_id = env.step(action_id)

    return tracker, decisions


# ---------------------------------------------------------------------------
# Token layout
# ---------------------------------------------------------------------------


def test_token_layout_uses_disjoint_one_hot_blocks():
    vector = encode_token(sample_token(), viewer=0)

    assert vector.shape == (TOKEN_DIM,)
    assert vector.dtype == np.float32
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)

    assert vector[ACTOR_SLICE].sum() == 1.0
    assert vector[ACTION_SLICE].sum() == 1.0
    assert vector[STREET_SLICE].sum() == 1.0

    assert vector[ACTOR_SLICE.start + 0] == 1.0  # P0
    assert vector[ACTION_SLICE.start + ACTION_NAMES.index("raise")] == 1.0
    assert vector[STREET_SLICE.start + 2] == 1.0  # turn
    assert vector[RAISES_INDEX] == pytest.approx(0.25)
    assert vector[POT_INDEX] == pytest.approx(6.5 / POT_CEILING_BB)
    assert vector[TO_CALL_INDEX] == pytest.approx(1.0 / CALL_CEILING_BB)
    assert vector[SMALL_BLIND_INDEX] == 1.0
    assert vector[IN_POSITION_INDEX] == 0.0


def test_is_self_depends_on_the_viewer():
    token = sample_token(actor=1)

    as_p0 = encode_token(token, viewer=0)
    as_p1 = encode_token(token, viewer=1)

    assert as_p1[IS_SELF_INDEX] == 1.0
    assert as_p0[IS_SELF_INDEX] == 0.0

    differing = np.flatnonzero(as_p0 != as_p1)
    assert differing.tolist() == [IS_SELF_INDEX]


def test_normalization_clips_instead_of_overflowing():
    vector = encode_token(
        sample_token(pot_bb=10_000.0, to_call_bb=10_000.0), viewer=0
    )

    assert vector[POT_INDEX] == 1.0
    assert vector[TO_CALL_INDEX] == 1.0


@pytest.mark.parametrize(
    "board_size, street",
    [(0, 0), (3, 1), (4, 2), (5, 3)],
)
def test_street_mapping(board_size, street):
    assert street_from_board_size(board_size) == street
    assert STREET_NAMES[street]


@pytest.mark.parametrize("board_size", [1, 2, 6, 7])
def test_unexpected_board_size_is_rejected(board_size):
    with pytest.raises(ValueError):
        street_from_board_size(board_size)


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor": 2},
        {"action": "bet"},
        {"street": 4},
        {"raises_this_street": 5},
    ],
)
def test_invalid_tokens_are_rejected(overrides):
    with pytest.raises(ValueError):
        encode_token(sample_token(**overrides), viewer=0)


def test_invalid_viewer_is_rejected():
    with pytest.raises(ValueError):
        encode_token(sample_token(), viewer=2)


# ---------------------------------------------------------------------------
# Empty history and hand lifecycle
# ---------------------------------------------------------------------------


def test_tracker_requires_start_hand(env):
    tracker = HistoryTracker()

    with pytest.raises(RuntimeError):
        tracker.tokens(viewer=0)

    state, _ = env.reset()
    with pytest.raises(RuntimeError):
        tracker.record(state, 0, "call")


def test_empty_history_is_all_false_mask(env):
    tracker = HistoryTracker()
    state, _ = env.reset()
    tracker.start_hand(state)

    sequence = tracker.tokens(viewer=0)

    assert sequence.length == 0
    assert sequence.tokens.shape == (MAX_HISTORY_TOKENS, TOKEN_DIM)
    assert not sequence.mask.any()
    assert np.count_nonzero(sequence.tokens) == 0
    assert sequence.real_tokens().shape == (0, TOKEN_DIM)


def test_first_decision_sees_no_history(env):
    tracker, decisions = play_tracked_hand(env, np.random.default_rng(3))

    assert decisions[0]["sequence"].length == 0
    assert decisions[0]["record_length"] == 0
    assert tracker.length == len(decisions)


def test_history_resets_between_hands(env):
    rng = np.random.default_rng(11)
    tracker, first_decisions = play_tracked_hand(env, rng)
    first_length = tracker.length
    assert first_length == len(first_decisions) > 0

    state, _ = env.reset()
    tracker.start_hand(state)

    assert tracker.length == 0
    assert tracker.dropped == 0
    assert not tracker.tokens(viewer=0).mask.any()


# ---------------------------------------------------------------------------
# Agreement with the environment
# ---------------------------------------------------------------------------


def test_token_stream_matches_action_record_order(env):
    tracker, decisions = play_tracked_hand(env, np.random.default_rng(7))

    recorded = [(int(pid), str(name)) for pid, name in env.action_recorder]
    tokens = [(token.actor, token.action) for token in tracker.actions]

    assert tokens == recorded
    assert [decision["record_length"] for decision in decisions] == list(
        range(len(decisions))
    )

    # At every decision the tracker held exactly the actions already taken.
    for position, decision in enumerate(decisions):
        assert decision["sequence"].length == position
        assert decision["sequence"].mask.sum() == position
        assert decision["sequence"].mask[:position].all()
        assert not decision["sequence"].mask[position:].any()


def test_pot_and_call_features_match_the_raw_state(env):
    rng = np.random.default_rng(19)

    for _ in range(10):
        tracker, decisions = play_tracked_hand(env, rng)
        big_blind = tracker.big_blind

        for token, decision in zip(tracker.actions, decisions):
            raw_obs = decision["raw_obs"]
            chips = raw_obs["all_chips"]
            assert token.pot_bb == pytest.approx(sum(chips) / big_blind)
            assert token.to_call_bb == pytest.approx(
                (max(chips) - raw_obs["my_chips"]) / big_blind
            )
            assert token.raises_this_street == decision["raises"][token.street]


def test_features_stay_normalized_across_many_hands(env):
    rng = np.random.default_rng(23)

    for _ in range(30):
        tracker, _ = play_tracked_hand(env, rng)
        sequence = tracker.tokens(viewer=0)
        real = sequence.real_tokens()

        assert real.shape[0] == sequence.length
        assert np.all(real >= 0.0)
        assert np.all(real <= 1.0)
        assert np.all(real[:, POT_INDEX] <= 1.0)
        assert np.all(real[:, TO_CALL_INDEX] <= 1.0)
        assert np.all(real[:, RAISES_INDEX] <= 1.0)


def test_small_blind_flag_matches_the_posted_blinds(env):
    rng = np.random.default_rng(31)
    seen_small_blinds = set()

    for _ in range(20):
        tracker, decisions = play_tracked_hand(env, rng)
        opening_chips = decisions[0]["raw_obs"]["all_chips"]

        assert tracker.sb_player == int(np.argmin(opening_chips))
        assert tracker.big_blind == max(opening_chips)
        seen_small_blinds.add(tracker.sb_player)

        for token in tracker.actions:
            assert token.is_small_blind == (token.actor == tracker.sb_player)

    # Blinds are drawn at random each hand, so both players must appear.
    assert seen_small_blinds == {0, 1}


def test_in_position_flag_alternates_within_each_street(env):
    rng = np.random.default_rng(37)

    for _ in range(10):
        tracker, _ = play_tracked_hand(env, rng)

        for street in range(len(STREET_NAMES)):
            on_street = [token for token in tracker.actions if token.street == street]
            actors = [token.actor for token in on_street]

            # Heads-up action order alternates, so acting second means acting last.
            assert all(
                later != earlier for earlier, later in zip(actors, actors[1:])
            )
            assert [
                token.is_in_position for token in on_street
            ] == [index % 2 == 1 for index in range(len(on_street))]


# ---------------------------------------------------------------------------
# Padding, masking and truncation
# ---------------------------------------------------------------------------


def test_truncation_keeps_the_most_recent_tokens(env):
    tracker = HistoryTracker(max_tokens=4)
    state, player_id = env.reset()
    tracker.start_hand(state)

    tokens = [
        sample_token(action="call", raises_this_street=index % (RAISE_LIMIT + 1))
        for index in range(6)
    ]
    for token in tokens:
        tracker.append(token)

    assert tracker.length == 4
    assert tracker.dropped == 2
    assert tracker.actions == tuple(tokens[-4:])

    sequence = tracker.tokens(viewer=0)
    assert sequence.length == 4
    assert sequence.mask.tolist() == [True, True, True, True]
    assert np.array_equal(
        sequence.real_tokens(), np.stack([encode_token(t, 0) for t in tokens[-4:]])
    )


def test_default_capacity_matches_the_constant(env):
    tracker = HistoryTracker()
    state, _ = env.reset()
    tracker.start_hand(state)

    for index in range(MAX_HISTORY_TOKENS + 3):
        tracker.append(sample_token(raises_this_street=index % 3))

    assert tracker.max_tokens == MAX_HISTORY_TOKENS
    assert tracker.length == MAX_HISTORY_TOKENS
    assert tracker.dropped == 3


def test_padding_rows_are_zero_and_masked_out(env):
    tracker = HistoryTracker(max_tokens=8)
    state, _ = env.reset()
    tracker.start_hand(state)
    tracker.append(sample_token())

    sequence = tracker.tokens(viewer=1)

    assert sequence.length == 1
    assert sequence.mask.tolist() == [True] + [False] * 7
    assert np.count_nonzero(sequence.tokens[1:]) == 0
    assert np.any(sequence.tokens[0] != 0)
