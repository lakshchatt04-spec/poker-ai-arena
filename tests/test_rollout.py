"""Tests for rollout collection and generalised advantage estimation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder
from poker_arena.environment import make_environment
from poker_arena.networks import NUM_ACTIONS, PolicyValueNet
from poker_arena.rollout import (
    GAE_LAMBDA,
    GAMMA,
    RolloutBuffer,
    Trajectory,
    Transition,
    collect_trajectories,
    compute_gae,
    normalize_advantages,
)

#: call and fold legal, raise and check not.  Action 0 is always legal here.
DEFAULT_MASK = np.array([True, False, True, False], dtype=bool)


def _transition(
    action: int = 0,
    *,
    mask: np.ndarray | None = None,
    value: float = 0.0,
    log_prob: float = 0.0,
    features: np.ndarray | None = None,
) -> Transition:
    if mask is None:
        mask = DEFAULT_MASK.copy()
    if features is None:
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
    return Transition(
        features=np.asarray(features, dtype=np.float32),
        legal_mask=np.asarray(mask, dtype=bool),
        action=int(action),
        log_prob=float(log_prob),
        value=float(value),
    )


def _trajectory(values, payoff: float, actions=None) -> Trajectory:
    values = list(values)
    if actions is None:
        actions = [0] * len(values)
    return Trajectory(
        transitions=tuple(
            _transition(action=action, value=value)
            for action, value in zip(actions, values)
        ),
        final_payoff=float(payoff),
    )


@pytest.fixture()
def env():
    return make_environment(seed=42)


@pytest.fixture()
def encoder(env):
    return ObservationEncoder.from_environment(env)


@pytest.fixture()
def net():
    torch.manual_seed(0)
    return PolicyValueNet()


# -- the GAE identity ------------------------------------------------------
def test_lambda_one_is_the_naive_terminal_return_minus_value():
    """lam=1 must reproduce R - V(s_t) exactly; that is the sanity anchor."""

    values = np.array([0.0, 1.0, 1.5])
    payoff = -3.0
    rewards = _trajectory(values, payoff).rewards()

    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=1.0)

    assert np.allclose(advantages, payoff - values, atol=1e-5)
    assert np.allclose(returns, np.full(3, payoff), atol=1e-5)


def test_lambda_zero_is_the_td_residual():
    """Each decision judged only by the value change it caused."""

    values = np.array([0.0, 1.0, 1.5])
    rewards = _trajectory(values, -3.0).rewards()

    advantages, _ = compute_gae(rewards, values, gamma=1.0, lam=0.0)

    # delta = [0 + 1.0 - 0, 0 + 1.5 - 1.0, -3.0 + 0 - 1.5]
    assert np.allclose(advantages, [1.0, 0.5, -4.5], atol=1e-5)


def test_worked_example_advantages_at_both_ends():
    """The table from the design discussion, as an assertion."""

    values = np.array([0.0, 1.0, 1.5])
    rewards = _trajectory(values, -3.0).rewards()

    at_zero, _ = compute_gae(rewards, values, lam=0.0)
    at_one, _ = compute_gae(rewards, values, lam=1.0)

    assert np.allclose(at_zero, [1.0, 0.5, -4.5], atol=1e-5)
    assert np.allclose(at_one, [-3.0, -4.0, -4.5], atol=1e-5)


def test_default_lambda_sits_between_the_two_ends():
    values = np.array([0.0, 1.0, 1.5])
    rewards = _trajectory(values, -3.0).rewards()

    at_zero, _ = compute_gae(rewards, values, lam=0.0)
    default, _ = compute_gae(rewards, values, lam=GAE_LAMBDA)
    at_one, _ = compute_gae(rewards, values, lam=1.0)

    assert GAMMA == 1.0
    assert GAE_LAMBDA == 0.95
    assert np.all(default >= at_one - 1e-6)
    assert np.all(default <= at_zero + 1e-6)


def test_returns_are_advantages_plus_values_for_every_lambda():
    values = np.array([-0.5, 0.25, -1.0, 2.0, 0.0])
    rewards = np.array([0.0, 0.0, 0.0, 0.0, 4.0])

    for lam in (0.0, 0.5, 0.95, 1.0):
        advantages, returns = compute_gae(rewards, values, lam=lam)
        assert np.allclose(returns, advantages + values.astype(np.float32), atol=1e-6)
        assert np.isfinite(advantages).all()
        assert np.isfinite(returns).all()


def test_terminal_transition_bootstraps_from_zero():
    """A losing last decision cannot inherit a phantom continuation value."""

    advantages, returns = compute_gae(
        np.array([-3.0]), np.array([1.0]), gamma=1.0, lam=0.0
    )

    assert advantages[0] == pytest.approx(-4.0, abs=1e-6)
    assert returns[0] == pytest.approx(-3.0, abs=1e-6)
    assert advantages[0] != pytest.approx(-1.0, abs=1e-6)


def test_a_tie_with_a_correct_value_has_no_signal():
    values = np.zeros(3)
    rewards = _trajectory(values, 0.0).rewards()

    advantages, returns = compute_gae(rewards, values, lam=0.95)

    assert np.allclose(advantages, 0.0)
    assert np.allclose(returns, 0.0)


def test_an_over_optimistic_value_is_corrected_by_the_terminal_payoff():
    """A flat 2 bb estimate on a hand that ties is blamed, not ignored."""

    values = np.full(3, 2.0)
    rewards = _trajectory(values, 0.0).rewards()

    advantages, returns = compute_gae(rewards, values, lam=GAE_LAMBDA)

    assert np.all(advantages < 0)
    assert advantages[-1] == pytest.approx(-2.0, abs=1e-6)
    assert np.allclose(returns, advantages + values.astype(np.float32), atol=1e-6)


def test_compute_gae_rejects_malformed_input():
    with pytest.raises(ValueError):
        compute_gae(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        compute_gae(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        compute_gae(np.zeros(0), np.zeros(0))


# -- normalisation ---------------------------------------------------------
def test_normalize_centres_and_scales():
    advantages = np.array([-4.0, 0.0, 1.0, 3.0], dtype=np.float32)

    normalised = normalize_advantages(advantages)

    assert normalised.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalised.std() == pytest.approx(1.0, abs=1e-4)
    assert np.isfinite(normalised).all()


def test_normalize_leaves_no_signal_as_zeros_instead_of_dividing_by_zero():
    assert np.allclose(normalize_advantages(np.full(5, 2.0)), 0.0)
    assert normalize_advantages(np.zeros(0)).shape == (0,)


# -- the buffer ------------------------------------------------------------
def test_buffer_flattens_trajectories_in_order():
    buffer = RolloutBuffer()
    first = Trajectory(
        transitions=(
            _transition(value=0.0, log_prob=-1.0,
                        features=np.full(FEATURE_DIM, 1.0, dtype=np.float32)),
            _transition(action=2, value=1.0, log_prob=-0.5,
                        features=np.full(FEATURE_DIM, 2.0, dtype=np.float32)),
        ),
        final_payoff=2.0,
    )
    second = Trajectory(
        transitions=(
            _transition(action=2, value=-1.0, log_prob=-0.2,
                        features=np.full(FEATURE_DIM, 3.0, dtype=np.float32)),
            _transition(value=0.0, log_prob=-0.1,
                        features=np.full(FEATURE_DIM, 4.0, dtype=np.float32)),
            _transition(action=2, value=1.0, log_prob=-0.3,
                        features=np.full(FEATURE_DIM, 5.0, dtype=np.float32)),
        ),
        final_payoff=-2.0,
    )

    buffer.add_trajectory(first)
    buffer.add_trajectory(second)
    batch = buffer.batch()

    assert len(buffer) == 5
    assert len(batch) == 5
    assert batch.features.shape == (5, FEATURE_DIM)
    assert batch.legal_masks.shape == (5, NUM_ACTIONS)
    assert batch.actions.shape == (5,)
    assert batch.log_probs.shape == (5,)
    assert batch.values.shape == (5,)
    assert batch.advantages.shape == (5,)
    assert batch.returns.shape == (5,)

    assert batch.features.dtype == np.float32
    assert batch.legal_masks.dtype == np.bool_
    assert batch.actions.dtype == np.int64

    assert np.allclose(batch.features[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert batch.actions.tolist() == [0, 2, 2, 0, 2]
    assert np.allclose(batch.values, [0.0, 1.0, -1.0, 0.0, 1.0])
    assert np.allclose(batch.log_probs, [-1.0, -0.5, -0.2, -0.1, -0.3])

    first_advantages, first_returns = compute_gae(
        first.rewards(), first.values(), gamma=GAMMA, lam=GAE_LAMBDA
    )
    assert np.allclose(batch.advantages[:2], first_advantages, atol=1e-6)
    assert np.allclose(batch.returns[:2], first_returns, atol=1e-6)
    assert np.allclose(batch.returns, batch.advantages + batch.values, atol=1e-6)


@pytest.mark.parametrize(
    "transition",
    [
        _transition(features=np.zeros(FEATURE_DIM - 1, dtype=np.float32)),
        _transition(mask=np.ones(NUM_ACTIONS - 1, dtype=bool)),
        Transition(
            features=np.zeros(FEATURE_DIM, dtype=np.float32),
            legal_mask=np.ones(NUM_ACTIONS, dtype=np.float32),
            action=0,
            log_prob=0.0,
            value=0.0,
        ),
        _transition(action=1),  # raise is illegal under DEFAULT_MASK
        _transition(action=NUM_ACTIONS),
        _transition(value=float("nan")),
        _transition(log_prob=float("inf")),
    ],
)
def test_buffer_rejects_invalid_transitions(transition):
    buffer = RolloutBuffer()
    with pytest.raises(ValueError):
        buffer.add_trajectory(Trajectory((transition,), final_payoff=0.0))


def test_buffer_rejects_empty_trajectories_and_empty_batches():
    buffer = RolloutBuffer()

    with pytest.raises(ValueError):
        buffer.add_trajectory(Trajectory((), final_payoff=0.0))
    with pytest.raises(ValueError):
        buffer.batch()

    buffer.add_trajectory(_trajectory([0.0], 1.0))
    assert len(buffer) == 1
    buffer.clear()
    assert len(buffer) == 0
    with pytest.raises(ValueError):
        buffer.batch()


def test_buffer_rejects_out_of_range_discounting():
    with pytest.raises(ValueError):
        RolloutBuffer(gamma=1.5)
    with pytest.raises(ValueError):
        RolloutBuffer(lam=-0.1)


# -- against the real game -------------------------------------------------
def test_collected_actions_are_legal_and_log_probs_reproduce(net, env, encoder):
    trajectories = collect_trajectories(
        env, encoder, net, hands=5, rng=np.random.default_rng(1)
    )

    assert trajectories
    checked = 0
    for trajectory in trajectories:
        assert len(trajectory) > 0
        for transition in trajectory.transitions:
            assert transition.legal_mask.dtype == np.bool_
            assert bool(transition.legal_mask[transition.action])
            assert transition.action in range(NUM_ACTIONS)
            assert np.isfinite(transition.value)
            assert np.isfinite(transition.log_prob)

            features = torch.from_numpy(transition.features).unsqueeze(0)
            mask = torch.from_numpy(transition.legal_mask).unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(features, mask)
                expected = Categorical(logits=logits).log_prob(
                    torch.tensor(transition.action)
                )
            assert transition.log_prob == pytest.approx(float(expected), abs=1e-5)
            checked += 1

    assert checked > 10


def test_self_play_payoffs_are_zero_sum_and_match_the_environment(net):
    """Both seats recorded: their payoffs are the environment's, and sum to 0."""

    found = False
    for offset in range(40):
        hand_env = make_environment(seed=200 + offset)
        hand_encoder = ObservationEncoder.from_environment(hand_env)
        trajectories = collect_trajectories(
            hand_env,
            hand_encoder,
            net,
            hands=1,
            rng=np.random.default_rng(offset),
        )
        if len(trajectories) != 2:
            continue

        payoffs = np.asarray(hand_env.get_payoffs(), dtype=float)
        assert trajectories[0].final_payoff == pytest.approx(payoffs[0])
        assert trajectories[1].final_payoff == pytest.approx(payoffs[1])
        assert trajectories[0].final_payoff + trajectories[1].final_payoff == (
            pytest.approx(0.0)
        )
        found = True
        break

    assert found, "no self-play hand recorded both seats in 40 attempts"


def test_a_callable_opponent_plays_player_one_and_only_player_zero_is_recorded(net):
    seen: list[int] = []

    def opponent(state, player_id):
        seen.append(player_id)
        return int(sorted(state["legal_actions"])[0])

    env = make_environment(seed=9)
    encoder = ObservationEncoder.from_environment(env)

    trajectories = collect_trajectories(
        env, encoder, net, opponent=opponent, hands=20,
        rng=np.random.default_rng(3),
    )

    assert seen, "the opponent was never asked to act"
    assert set(seen) == {1}
    assert len(trajectories) <= 20
    assert all(len(trajectory) > 0 for trajectory in trajectories)


def test_learner_payoff_matches_player_zero(net):
    def opponent(state, player_id):
        return int(sorted(state["legal_actions"])[0])

    for offset in range(20):
        env = make_environment(seed=300 + offset)
        encoder = ObservationEncoder.from_environment(env)
        trajectories = collect_trajectories(
            env, encoder, net, opponent=opponent, hands=1,
            rng=np.random.default_rng(offset),
        )
        if not trajectories:
            continue
        assert trajectories[0].final_payoff == pytest.approx(
            float(env.get_payoffs()[0])
        )
        return

    pytest.fail("player 0 never had a decision in 20 hands")


def test_invalid_opponent_action_raises(net, env, encoder):
    with pytest.raises(ValueError, match="illegal action"):
        collect_trajectories(
            env, encoder, net, opponent=lambda state, player: 99, hands=20,
            rng=np.random.default_rng(0),
        )


def test_hands_must_be_positive(net, env, encoder):
    with pytest.raises(ValueError):
        collect_trajectories(env, encoder, net, hands=0)


def test_same_seeds_give_identical_rollouts():
    def run():
        env = make_environment(seed=11)
        run_encoder = ObservationEncoder.from_environment(env)
        torch.manual_seed(0)
        run_net = PolicyValueNet()
        return collect_trajectories(
            env, run_encoder, run_net, hands=3,
            rng=np.random.default_rng(5),
        )

    first = run()
    second = run()

    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left.final_payoff == right.final_payoff
        assert len(left) == len(right)
        for a, b in zip(left.transitions, right.transitions):
            assert a.action == b.action
            assert a.value == pytest.approx(b.value)
            assert a.log_prob == pytest.approx(b.log_prob)
            assert np.array_equal(a.features, b.features)
            assert np.array_equal(a.legal_mask, b.legal_mask)


def test_collection_restores_the_networks_training_mode(net, env, encoder):
    net.train()
    collect_trajectories(env, encoder, net, hands=1, rng=np.random.default_rng(0))
    assert net.training

    net.eval()
    collect_trajectories(env, encoder, net, hands=1, rng=np.random.default_rng(0))
    assert not net.training


def test_a_hand_can_leave_a_recorded_player_with_no_decisions(net):
    """If the small blind folds preflop, the big blind never acts: skip, not crash."""

    empty_hands = 0
    recorded_hands = 0

    for offset in range(60):
        env = make_environment(seed=500 + offset)
        encoder = ObservationEncoder.from_environment(env)

        trajectories = collect_trajectories(
            env,
            encoder,
            net,
            opponent=lambda state, player: 2,  # fold, which is always legal
            hands=1,
            rng=np.random.default_rng(offset),
        )

        if not trajectories:
            empty_hands += 1
        else:
            recorded_hands += 1
            assert len(trajectories) == 1
            assert len(trajectories[0]) > 0

    assert empty_hands > 0, "no hand left player 0 with zero decisions"
    assert recorded_hands > 0
