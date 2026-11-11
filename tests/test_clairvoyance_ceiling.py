"""Tests for the clairvoyance-ceiling experiment in ``scripts/clairvoyance_ceiling.py``.

The script's claim rests on a fork being exact: the replay must start from the
same observation, the same raise counts, the same deck and the same history
tokens as the deal it came from.  A policy that silently drifts after the fork
would make every ceiling estimate meaningless, so the natural-runout replay of
the actual action is pinned to the hand's real payoff here.

The estimator is separated from the simulator so it can be checked on synthetic
value tables: cross-fitting must score the selected action only on runouts the
selector did not see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rlcard.games.base import Card

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import clairvoyance_ceiling as cc

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker


class _FoldPolicy:
    """Fold at the first decision, so the hand ends before the flop."""

    def __call__(self, state, player_id, tracker, small_blind):
        legal = set(state["legal_actions"])
        return 2 if 2 in legal else int(min(legal))


class _ComplementPolicy:
    """Call a bot raise and raise a bot call, then check/call.

    This is the response ``coin_opener`` is designed to collide with: both coin
    values reach the agent's first flop decision with the same chips and raise
    counts, so the private coin is only in the action record.
    """

    def __call__(self, state, player_id, tracker, small_blind):
        legal = set(state["legal_actions"])
        raw = state["raw_obs"] or {}
        if not raw.get("public_cards"):
            raises = int(raw.get("raise_nums", [0])[0])
            if raises >= 1:
                for action in (0, 3, 2):
                    if action in legal:
                        return action
            elif 1 in legal:
                return 1
        for action in (3, 0, 2):
            if action in legal:
                return action
        return int(min(legal))


def _fake_snapshot(deck, **overrides):
    fields = dict(
        hand=0,
        deal_seed=1,
        coin=0,
        steps=0,
        small_blind=0,
        obs=np.zeros(1, dtype=np.float32),
        legal_actions=(1, 2, 3),
        raise_nums=(0, 0, 0, 0),
        action_recorder_length=0,
        tracker=HistoryTracker(),
        deck=tuple(deck),
        natural_runout=(str(deck[-1].get_index()), str(deck[-2].get_index())),
    )
    fields.update(overrides)
    return cc.ForkSnapshot(**fields)


def _small_deck():
    return [Card("S", rank) for rank in "234567"]


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


def test_ceiling_is_zero_when_the_actual_action_is_always_best():
    table = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    estimates = cc.ceiling_estimates(table, actual_row=0)
    assert estimates.cross_fitted == pytest.approx(0.0)
    assert estimates.in_sample == pytest.approx(0.0)
    assert estimates.per_runout == pytest.approx(0.0)
    assert estimates.best_row == 0


def test_ceiling_is_the_gap_when_a_dominated_action_is_better():
    table = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    estimates = cc.ceiling_estimates(table, actual_row=0)
    assert estimates.cross_fitted == pytest.approx(1.0)
    assert estimates.in_sample == pytest.approx(1.0)
    assert estimates.per_runout == pytest.approx(1.0)
    assert estimates.best_row == 1


def test_cross_fitting_does_not_score_the_selector_on_its_own_runouts():
    """A spurious winner on one half must lose on the other half."""

    table = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, -1.0, -1.0],
        ]
    )
    estimates = cc.ceiling_estimates(table, actual_row=0)
    # Selection on columns 0-1 picks row 1, which scores -1 on columns 2-3;
    # selection on columns 2-3 correctly keeps row 0.  The average is -0.5.
    assert estimates.cross_fitted == pytest.approx(-0.5)
    # In-sample, both rows average zero and the tie goes to the first row.
    assert estimates.in_sample == pytest.approx(0.0)


def test_ceiling_rejects_tables_it_cannot_cross_fit():
    with pytest.raises(ValueError):
        cc.ceiling_estimates(np.zeros((2, 1)), actual_row=0)
    with pytest.raises(ValueError):
        cc.ceiling_estimates(np.zeros((2, 4)), actual_row=2)


def test_per_runout_ceiling_includes_foresight():
    table = np.asarray(
        [
            [0.0, 0.0],
            [2.0, -1.0],
            [-1.0, 3.0],
        ]
    )
    estimates = cc.ceiling_estimates(table, actual_row=0)
    assert estimates.per_runout == pytest.approx((2.0 + 3.0) / 2.0)
    assert estimates.in_sample == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Deck and runouts
# --------------------------------------------------------------------------


def test_deck_with_runout_deals_turn_then_river():
    deck = _small_deck()
    snapshot = _fake_snapshot(deck)
    ordered = cc.deck_with_runout(snapshot, "S3", "S5")
    turn = ordered.pop()
    river = ordered.pop()
    assert turn.get_index() == "S3"
    assert river.get_index() == "S5"
    assert len(ordered) == len(deck) - 2
    assert {card.get_index() for card in ordered}.isdisjoint({"S3", "S5"})


def test_deck_with_runout_rejects_cards_not_in_the_deck():
    snapshot = _fake_snapshot(_small_deck())
    with pytest.raises(ValueError):
        cc.deck_with_runout(snapshot, "SA", "S5")
    with pytest.raises(ValueError):
        cc.deck_with_runout(snapshot, "S3", "S3")


def test_sample_runouts_are_distinct_and_come_from_the_deck():
    snapshot = _fake_snapshot(_small_deck())
    rng = np.random.default_rng(0)
    pairs = cc.sample_runouts(snapshot, 5, rng)
    assert len(pairs) == len(set(pairs)) == 5
    available = {str(card.get_index()) for card in snapshot.deck}
    for pair in pairs:
        assert len(set(pair)) == 2
        assert set(pair) <= available


# --------------------------------------------------------------------------
# Card-free grouping
# --------------------------------------------------------------------------


def _features(**changes):
    values = np.zeros(FEATURE_DIM, dtype=np.float32)
    values[cc.BETTING_SLICE] = 0.25
    for index, value in changes.items():
        values[int(index)] = value
    return values


def test_betting_key_ignores_cards_but_not_betting():
    first = _features()
    second = _features()
    second[0] = 1.0
    second[51] = 1.0
    assert cc._betting_key(first) == cc._betting_key(second)

    third = _features()
    third[FEATURE_DIM - 1] = 0.75
    assert cc._betting_key(first) != cc._betting_key(third)


def test_mixed_keys_need_both_coins_under_the_same_summary():
    key = b"shared"
    other = b"only-heads"
    records = [
        cc.FirstFlopRecord(hand=0, deal_seed=1, flat_key=key, coin=0, payoff=0.0),
        cc.FirstFlopRecord(hand=1, deal_seed=2, flat_key=key, coin=1, payoff=0.0),
        cc.FirstFlopRecord(hand=2, deal_seed=3, flat_key=other, coin=1, payoff=0.0),
    ]
    assert cc.mixed_keys(records) == {key}


# --------------------------------------------------------------------------
# Environment fork
# --------------------------------------------------------------------------


def test_hand_that_ends_preflop_has_no_first_flop_fork():
    env = make_environment(seed=10_000, allow_step_back=True)
    opponent = cc.make_opponent("coin_opener", seed=10_000)
    tracker = HistoryTracker()
    assert cc.play_to_first_flop(env, _FoldPolicy(), opponent, tracker, 10_000) is None


def test_fork_restores_the_observation_and_the_raise_block():
    env = make_environment(seed=10_000, allow_step_back=True)
    policy = _ComplementPolicy()
    tracker = HistoryTracker()
    deal_seed = 10_000
    opponent = cc.make_opponent("coin_opener", seed=deal_seed)

    fork = cc.play_to_first_flop(env, policy, opponent, tracker, deal_seed)
    assert fork is not None
    state, player, small_blind, steps = fork
    assert player == 0
    assert len(state["raw_obs"]["public_cards"]) == 3
    assert steps == len(env.game.history)

    snapshot = cc.make_snapshot(
        env, state, tracker,
        hand=0, deal_seed=deal_seed, coin=int(opponent.coin),
        steps=steps, small_blind=small_blind,
    )
    action = snapshot.legal_actions[0]
    next_state, next_player = env.step(action)
    cc.play_out(env, policy, opponent, tracker, next_state, next_player, small_blind)
    restored = cc.restore(env, snapshot)

    assert np.array_equal(restored["obs"], snapshot.obs)
    assert tuple(sorted(restored["legal_actions"])) == snapshot.legal_actions
    assert env.game.get_player_id() == 0
    assert tuple(env.game.history_raise_nums) == snapshot.raise_nums
    assert len(env.action_recorder) == snapshot.action_recorder_length
    assert len(env.game.history) == snapshot.steps


def test_natural_replay_reproduces_the_actual_hand_payoff():
    base_seed = 30_000
    hands = 12
    env_play = make_environment(seed=base_seed)
    encoder = ObservationEncoder.from_environment(env_play)
    policy = _ComplementPolicy()

    records = cc.collect_first_flops(
        env_play, encoder, policy, "coin_opener", hands, base_seed
    )
    assert records, "the complement policy should reach the flop on some deals"
    wanted = cc.mixed_keys(records)
    assert wanted, "the two coin values should collide at some first flops"

    env_fork = make_environment(seed=base_seed, allow_step_back=True)
    results = cc.evaluate_records(
        env_fork, encoder, policy, "coin_opener", records, wanted, 4, base_seed
    )
    # ``evaluate_records`` raises when the natural replay of the actual action
    # disagrees with the hand's payoff, which is the invariant this test pins.
    assert len(results) == sum(record.flat_key in wanted for record in records)
    for result in results:
        row = result.legal_actions.index(result.actual_action)
        assert result.q_natural[row] != pytest.approx(float("nan"))
