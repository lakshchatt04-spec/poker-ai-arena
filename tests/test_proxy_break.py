"""Tests for the proxy-breaking diagnostic in ``scripts/proxy_break.py``.

The point of the script is a counterfactual: the forced opening must be
*identical* for both policies on the same deal, otherwise the paired
history-on/history-zeroed comparison is not measuring the history channel.  The
speed of the script matters less than that invariant, so it is pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import proxy_break

from poker_arena.environment import make_environment, reset_hand
from poker_arena.history import HistoryTracker
from poker_arena.training import make_opponent


class _CallPolicy:
    """Always check/call when possible."""

    def __call__(self, state, player_id, tracker, small_blind):
        legal = set(state["legal_actions"])
        for action in (3, 0, 2):
            if action in legal:
                return action
        return int(min(legal))


class _RaisePolicy:
    """Always raise when possible."""

    def __call__(self, state, player_id, tracker, small_blind):
        legal = set(state["legal_actions"])
        if 1 in legal:
            return 1
        for action in (3, 0, 2):
            if action in legal:
                return action
        return int(min(legal))


def test_forced_opening_only_returns_legal_actions():
    state = {"legal_actions": [0, 1, 2]}
    for seed in range(20):
        action = proxy_break._forced_opening(
            state, np.random.default_rng(seed)
        )
        assert action in state["legal_actions"]


class _AlwaysRaiseRng:
    """Minimal rng stub whose coin always votes for a raise."""

    def random(self) -> float:
        return 0.0


def test_forced_opening_raises_when_the_coin_says_so():
    assert proxy_break._forced_opening(
        {"legal_actions": [0, 1, 2, 3]}, _AlwaysRaiseRng()
    ) == 1


def test_forced_opening_falls_back_when_raise_is_illegal():
    state = {"legal_actions": [0, 3]}
    # An rng that always votes for a raise must not violate legality.
    assert proxy_break._forced_opening(state, _AlwaysRaiseRng()) == 3


def test_hole_strength_bounds_and_order():
    assert proxy_break._hole_strength(["SA", "HA"]) == pytest.approx(1.0)
    assert proxy_break._hole_strength(["S2", "H3"]) == pytest.approx(5 / 28)
    assert proxy_break._hole_strength(["SA", "HA"]) > proxy_break._hole_strength(
        ["S7", "H2"]
    )


def test_summary_matches_the_confidence_interval():
    values = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    mean, half = proxy_break._summary(values)
    assert mean == pytest.approx(0.2)
    assert half == pytest.approx(
        proxy_break.Z_95 * values.std(ddof=1) / np.sqrt(3)
    )


def test_quarter_raise_rates_separate_weak_from_strong():
    strength = np.linspace(0.0, 1.0, 20)
    opening = (strength >= 0.5).astype(np.float64)
    weak, strong = proxy_break._quarter_raise_rates(strength, opening)
    assert weak == pytest.approx(0.0)
    assert strong == pytest.approx(1.0)


def test_forced_opening_is_policy_independent_on_the_same_deal():
    """Different policies must still see the same forced opening."""

    env = make_environment(seed=10_000)
    tracker = HistoryTracker()
    deal_seed = 10_017

    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)
    _, call_info = proxy_break._play_hand(
        env, _CallPolicy(), opponent, tracker, forced=True, deal_seed=deal_seed
    )

    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)
    _, raise_info = proxy_break._play_hand(
        env, _RaisePolicy(), opponent, tracker, forced=True, deal_seed=deal_seed
    )

    assert call_info["agent_acted_preflop"] == 1.0
    assert raise_info["agent_acted_preflop"] == 1.0
    assert call_info["opening_raise"] == raise_info["opening_raise"]


def test_play_hand_returns_a_finite_payoff():
    env = make_environment(seed=10_000)
    tracker = HistoryTracker()
    deal_seed = 10_003
    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)

    payoff, info = proxy_break._play_hand(
        env, _CallPolicy(), opponent, tracker, forced=False, deal_seed=deal_seed
    )

    assert np.isfinite(payoff)
    assert info["agent_acted_preflop"] in (0.0, 1.0)
    assert 0.0 <= info["strength"] <= 1.0


class _DistributionPolicy:
    """Policy stub with a fixed opening distribution."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def action_probs(self, state, player_id, tracker, small_blind):
        return self.probs


def test_sample_opening_respects_the_distribution_support():
    policy = _DistributionPolicy([0.0, 1.0, 0.0, 0.0])
    for seed in range(20):
        action = proxy_break._sample_opening(
            policy, {"legal_actions": [0, 1]}, 0, None, 0,
            np.random.default_rng(seed),
        )
        assert action == 1


def test_sampled_opening_is_identical_for_both_arms_on_the_same_deal():
    env = make_environment(seed=10_000)
    tracker = HistoryTracker()
    deal_seed = 10_013

    def sampler(state, player_id, tracker, small_blind):
        return 1 if 1 in state["legal_actions"] else 0

    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)
    _, sampled_info = proxy_break._play_hand(
        env, _CallPolicy(), opponent, tracker,
        forced=False, deal_seed=deal_seed, sampler=sampler,
    )

    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)
    _, replayed_info = proxy_break._play_hand(
        env, _RaisePolicy(), opponent, tracker,
        forced=False, deal_seed=deal_seed,
        sampled_action=int(sampled_info["opening_action"]),
    )

    assert sampled_info["agent_acted_preflop"] == 1.0
    assert replayed_info["agent_acted_preflop"] == 1.0
    assert sampled_info["opening_action"] == replayed_info["opening_action"]
    assert sampled_info["opening_raise"] == replayed_info["opening_raise"]


def test_illegal_sampled_opening_is_rejected():
    env = make_environment(seed=10_000)
    tracker = HistoryTracker()
    deal_seed = 10_021
    env.seed(deal_seed)
    opponent = make_opponent("passive", seed=deal_seed)

    with pytest.raises(ValueError, match="illegal"):
        proxy_break._play_hand(
            env, _CallPolicy(), opponent, tracker,
            forced=False, deal_seed=deal_seed, sampled_action=1_234,
        )
