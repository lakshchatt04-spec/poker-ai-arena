"""Tests for the history-dependent opponent and the observation collision it uses.

The collision test is the important one: it asserts on real RLCard states that two
different action sequences produce the *same* encoded feature vector, which is the
precondition for a memory test.  The rest pins the opponent's per-hand state
machine and the hook the play loops use to reset it.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_arena.encoding import ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment, reset_hand
from poker_arena.opponents import (
    ContinuationBettor,
    FirstRaiserBettor,
    notify_hand_start,
)
from poker_arena.training import OPPONENT_NAMES, make_opponent


def _preflop_state() -> dict:
    return {"legal_actions": {0, 1, 2}, "raw_obs": {"public_cards": []}}


def _flop_state(*, raise_legal: bool = True) -> dict:
    legal = {0, 3} | ({1} if raise_legal else set())
    return {"legal_actions": legal, "raw_obs": {"public_cards": ["S9", "C9", "CK"]}}


def test_first_preflop_decision_is_a_coin_flip():
    raiser = ContinuationBettor(seed=0, preflop_raise_probability=1.0)
    raiser.start_hand(_preflop_state())
    assert raiser(_preflop_state(), 1) == 1
    assert raiser.was_preflop_aggressor is True

    caller = ContinuationBettor(seed=0, preflop_raise_probability=0.0)
    caller.start_hand(_preflop_state())
    assert caller(_preflop_state(), 1) == 0
    assert caller.was_preflop_aggressor is False


def test_mode_is_decided_once_per_hand_and_only_preflop():
    bot = ContinuationBettor(seed=3, preflop_raise_probability=1.0)
    bot.start_hand(_preflop_state())
    assert bot(_preflop_state(), 1) == 1
    # A second preflop decision must not re-roll, and must not raise again.
    assert bot(_flop_state() | {"raw_obs": {"public_cards": []}}, 1) != 1


def test_start_hand_clears_the_mode():
    bot = ContinuationBettor(seed=1, preflop_raise_probability=1.0)
    bot.start_hand(_preflop_state())
    bot(_preflop_state(), 1)
    assert bot.was_preflop_aggressor is True

    bot.start_hand(_preflop_state())
    assert bot.was_preflop_aggressor is False


def test_aggressor_bets_and_non_aggressor_does_not():
    aggressor = ContinuationBettor(seed=0, preflop_raise_probability=1.0)
    aggressor.start_hand(_preflop_state())
    aggressor(_preflop_state(), 1)
    assert aggressor(_flop_state(), 1) == 1

    passive = ContinuationBettor(seed=0, preflop_raise_probability=0.0)
    passive.start_hand(_preflop_state())
    passive(_preflop_state(), 1)
    assert passive(_flop_state(), 1) in (0, 3)


def test_make_opponent_knows_the_name():
    assert "c_bettor" in OPPONENT_NAMES
    opponent = make_opponent("c_bettor", seed=5)
    assert isinstance(opponent, ContinuationBettor)


def test_notify_hand_start_is_optional():
    calls = []

    class Stateful:
        def start_hand(self, state):
            calls.append(state)

        def __call__(self, state, player_id):
            return 0

    state = _preflop_state()
    notify_hand_start(Stateful(), state)
    assert calls == [state]

    notify_hand_start(lambda state, player_id: 0, state)  # must not raise


def _play_preflop(engine_seed: int, actions: tuple[int, ...]):
    env = make_environment(seed=engine_seed)
    encoder = ObservationEncoder.from_environment(env)
    state, player_id = reset_hand(env)
    small_blind = small_blind_player(state)
    for action in actions:
        state, player_id = env.step(action)
    return state, player_id, small_blind, encoder


@pytest.mark.parametrize("engine_seed", [7, 11, 23])
def test_observation_collides_across_preflop_sequences(engine_seed):
    """Two different preflop lines reach the same encoded flop state.

    Small blind raises and big blind calls, versus small blind calls, big blind
    raises and small blind calls.  Each player ends with 4 chips and the raise
    count is 1.  The big blind is the acting player in both, so ``is_small_blind``
    is ``False`` for both.  The action records differ, which is the only place
    the role information lives.
    """

    raised_then_called = _play_preflop(engine_seed, (1, 0, 3))
    called_then_raised = _play_preflop(engine_seed, (0, 1, 0))

    state_a, player_a, small_blind_a, encoder_a = raised_then_called
    state_b, player_b, small_blind_b, encoder_b = called_then_raised

    # RLCard draws the small blind at random, so derive the big blind from it
    # rather than assuming player 0 is out of position.
    big_blind = 1 - small_blind_a
    assert small_blind_a == small_blind_b
    assert player_a == big_blind == player_b
    assert state_a["raw_obs"]["all_chips"] == state_b["raw_obs"]["all_chips"]
    assert state_a["raw_obs"]["raise_nums"] == state_b["raw_obs"]["raise_nums"]
    assert state_a["raw_obs"]["public_cards"] == state_b["raw_obs"]["public_cards"]

    features_a = encoder_a.encode(state_a, is_small_blind=False).features
    features_b = encoder_b.encode(state_b, is_small_blind=False).features
    np.testing.assert_array_equal(features_a, features_b)

    assert state_a["action_record"] != state_b["action_record"]


def _first_raiser_flop_state(raises: int = 1, *, raise_legal: bool = True) -> dict:
    legal = {0, 3} | ({1} if raise_legal else set())
    return {
        "legal_actions": legal,
        "raw_obs": {
            "public_cards": ["S9", "C9", "CK"],
            "raise_nums": [raises, 0, 0, 0],
        },
    }


def test_first_raiser_preflop_coin_flip():
    raiser = FirstRaiserBettor(seed=0, preflop_raise_probability=1.0)
    raiser.start_hand(_preflop_state())
    assert raiser(_preflop_state(), 1) == 1
    # "Who opened" is only knowable once the flop is out.
    assert raiser.agent_raised_first is None

    caller = FirstRaiserBettor(seed=0, preflop_raise_probability=0.0)
    caller.start_hand(_preflop_state())
    assert caller(_preflop_state(), 1) == 0
    assert caller.agent_raised_first is None


def test_first_raiser_mode_depends_on_who_opened_preflop():
    agent_opened = FirstRaiserBettor(seed=0, preflop_raise_probability=0.0)
    agent_opened.start_hand(_preflop_state())
    assert agent_opened(_preflop_state(), 1) == 0
    assert agent_opened(_first_raiser_flop_state(raises=1), 1) == 1
    assert agent_opened.agent_raised_first is True

    bot_opened = FirstRaiserBettor(seed=0, preflop_raise_probability=1.0)
    bot_opened.start_hand(_preflop_state())
    assert bot_opened(_preflop_state(), 1) == 1
    assert bot_opened(_first_raiser_flop_state(raises=1), 1) != 1
    assert bot_opened.agent_raised_first is False


def test_first_raiser_without_a_preflop_raise_never_bets():
    bot = FirstRaiserBettor(seed=0, preflop_raise_probability=0.0)
    bot.start_hand(_preflop_state())
    bot(_preflop_state(), 1)
    assert bot(_first_raiser_flop_state(raises=0), 1) != 1
    assert bot.agent_raised_first is False


def test_first_raiser_mode_is_decided_once_per_hand():
    bot = FirstRaiserBettor(seed=0, preflop_raise_probability=0.0)
    bot.start_hand(_preflop_state())
    bot(_preflop_state(), 1)
    assert bot(_first_raiser_flop_state(raises=1), 1) == 1
    # A later street cannot flip the mode even if the current raise count looks
    # as if nobody opened.
    assert bot(_first_raiser_flop_state(raises=0), 1) == 1
    assert bot.agent_raised_first is True


def test_first_raiser_start_hand_clears_the_mode():
    bot = FirstRaiserBettor(seed=0, preflop_raise_probability=0.0)
    bot.start_hand(_preflop_state())
    bot(_preflop_state(), 1)
    bot(_first_raiser_flop_state(raises=1), 1)
    assert bot.agent_raised_first is True

    bot.start_hand(_preflop_state())
    assert bot.agent_raised_first is None


def test_make_opponent_knows_the_first_raiser_name():
    assert "first_raiser" in OPPONENT_NAMES
    opponent = make_opponent("first_raiser", seed=5)
    assert isinstance(opponent, FirstRaiserBettor)


def _play_to_agent_flop_decision(engine_seed: int, *, bot_raises_preflop: bool):
    """Reach the big blind's first flop decision with the bot on the small blind.

    ``bot_raises_preflop=False`` plays bot call, agent raise, bot call.
    ``bot_raises_preflop=True`` plays bot raise, agent call, bot flop check.
    The returned action is the bot's first post-flop action: a bet in the first
    line (the agent opened preflop) and a check in the second.
    """

    env = make_environment(seed=engine_seed)
    encoder = ObservationEncoder.from_environment(env)
    state, player_id = reset_hand(env)
    assert small_blind_player(state) == 1, "these seeds must put the small blind on player 1"

    bot = FirstRaiserBettor(
        seed=engine_seed,
        preflop_raise_probability=1.0 if bot_raises_preflop else 0.0,
    )
    notify_hand_start(bot, state)

    agent = 0
    agent_preflop_action = 0 if bot_raises_preflop else 1
    bot_postflop_action = None

    while not env.is_over():
        flop_dealt = bool(state["raw_obs"]["public_cards"])
        if player_id == agent and flop_dealt:
            break
        if player_id == agent:
            action = agent_preflop_action
        else:
            action = int(bot(state, player_id))
            if flop_dealt:
                bot_postflop_action = action
        state, player_id = env.step(action)

    features = encoder.encode(state, is_small_blind=False).features
    if bot_postflop_action is None:
        bot_postflop_action = int(bot(state, 1))
    return state, features, bot, bot_postflop_action


@pytest.mark.parametrize("engine_seed", [1, 4, 7, 9, 12, 13, 18, 20, 21, 22])
def test_first_raiser_hides_who_opened_at_the_agents_flop_decision(engine_seed):
    """The same features hide opposite bot behaviour.

    The agent opened preflop and the bot will bet the flop, or the bot opened
    preflop and has already checked.  Cards, chips, raise counts and street are
    identical, so only the action order (the history) tells them apart.
    """

    agent_opened = _play_to_agent_flop_decision(
        engine_seed, bot_raises_preflop=False
    )
    bot_opened = _play_to_agent_flop_decision(
        engine_seed, bot_raises_preflop=True
    )

    state_a, features_a, bot_a, action_a = agent_opened
    state_b, features_b, bot_b, action_b = bot_opened

    assert state_a["raw_obs"]["all_chips"] == state_b["raw_obs"]["all_chips"]
    assert state_a["raw_obs"]["raise_nums"] == state_b["raw_obs"]["raise_nums"]
    assert state_a["raw_obs"]["public_cards"] == state_b["raw_obs"]["public_cards"]
    np.testing.assert_array_equal(features_a, features_b)

    assert state_a["action_record"] != state_b["action_record"]
    assert bot_a.agent_raised_first is True
    assert bot_b.agent_raised_first is False
    assert action_a == 1
    assert action_b != 1
