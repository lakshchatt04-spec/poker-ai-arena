"""Tests for Agent A's policy/value network."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from poker_arena.encoding import (
    FEATURE_DIM,
    ObservationEncoder,
    small_blind_player,
)
from poker_arena.environment import make_environment
from poker_arena.networks import (
    MASKED_LOGIT,
    NUM_ACTIONS,
    PolicyValueNet,
)

EXPECTED_PARAMETERS = 100_869


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


def _real_batch(env, encoder, count: int):
    """Collect ``count`` encoded decision states from real hands."""

    rng = np.random.default_rng(0)
    features, masks = [], []
    while len(features) < count:
        state, player_id = env.reset()
        small_blind = small_blind_player(state)
        while not env.is_over() and len(features) < count:
            encoded = encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            features.append(encoded.features)
            masks.append(encoded.legal_mask)
            legal = list(state["legal_actions"])
            state, player_id = env.step(int(rng.choice(legal)))
    return torch.from_numpy(np.stack(features)), torch.from_numpy(np.stack(masks))


# -- wiring ---------------------------------------------------------------
def test_output_shapes_and_dtype(net):
    for batch in (1, 7, 256):
        features = torch.zeros(batch, FEATURE_DIM)
        legal_mask = torch.ones(batch, NUM_ACTIONS, dtype=torch.bool)

        logits, value = net(features, legal_mask)

        assert logits.shape == (batch, NUM_ACTIONS)
        assert value.shape == (batch,)
        assert logits.dtype == torch.float32
        assert value.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()


def test_parameter_count_is_locked(net):
    """Locks the architecture: 131 -> 256 -> 256 -> {4, 1}."""

    assert net.count_parameters() == EXPECTED_PARAMETERS
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMETERS
    assert sum(p.numel() for p in net.trunk.parameters()) == 99_584
    assert sum(p.numel() for p in net.policy_head.parameters()) == 1_028
    assert sum(p.numel() for p in net.value_head.parameters()) == 257


def test_batch_rows_match_single_sample_rows(net, env, encoder):
    """No cross-talk: row i of a batch equals that row scored alone."""

    features, masks = _real_batch(env, encoder, 8)
    logits, value = net(features, masks)

    for row in range(features.shape[0]):
        single_logits, single_value = net(features[row : row + 1], masks[row : row + 1])
        assert torch.allclose(logits[row], single_logits[0], atol=1e-5)
        assert torch.allclose(value[row], single_value[0], atol=1e-5)


def test_wrong_input_shape_is_rejected(net):
    legal_mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)

    with pytest.raises(ValueError):
        net(torch.zeros(2, FEATURE_DIM - 1), legal_mask)
    with pytest.raises(ValueError):
        net(torch.zeros(FEATURE_DIM), torch.ones(NUM_ACTIONS, dtype=torch.bool))


def test_different_states_produce_different_logits(net, env, encoder):
    features, masks = _real_batch(env, encoder, 2)
    logits, _ = net(features, masks)

    assert not torch.equal(logits[0], logits[1])
    assert not torch.equal(net.trunk(features)[0], net.trunk(features)[1])


def test_one_backward_reaches_every_parameter(net, env, encoder):
    features, _ = _real_batch(env, encoder, 1)
    legal_mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)

    logits, value = net(features, legal_mask)
    (logits.sum() + value.pow(2).sum()).backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is exactly zero"


def test_invalid_constructor_arguments_are_rejected():
    with pytest.raises(ValueError):
        PolicyValueNet(feature_dim=0)
    with pytest.raises(ValueError):
        PolicyValueNet(hidden=0)
    with pytest.raises(ValueError):
        PolicyValueNet(num_actions=NUM_ACTIONS - 1)


# -- masking --------------------------------------------------------------
def test_illegal_logits_are_exactly_the_mask_value(net, env, encoder):
    features, masks = _real_batch(env, encoder, 6)
    logits, _ = net(features, masks)

    unmasked = net.policy_head(net.trunk(features))

    assert torch.all(masks.any(dim=1))  # every row has something legal
    assert not bool(masks.all())  # and the batch does contain illegal actions
    assert torch.equal(
        logits[~masks], torch.full_like(logits[~masks], MASKED_LOGIT)
    )
    assert torch.equal(logits[masks], unmasked[masks])


def test_illegal_actions_get_zero_probability(net, env, encoder):
    features, masks = _real_batch(env, encoder, 6)
    logits, _ = net(features, masks)

    probabilities = torch.softmax(logits, dim=-1)

    assert torch.all(probabilities[~masks] == 0.0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(6))


def test_capped_state_never_samples_the_illegal_raise(net):
    """The failure mode that folds a hand: sampling raise where raising is capped."""

    features = torch.zeros(1, FEATURE_DIM)
    legal_mask = torch.tensor([[True, False, True, False]])  # call, fold

    logits, _ = net(features, legal_mask)
    samples = Categorical(logits=logits).sample((10_000,))

    assert not bool((samples == 1).any())
    assert set(samples.unique().tolist()) <= {0, 2}


def test_log_prob_and_entropy_are_finite(net, env, encoder):
    """The -inf lesson: entropy must not be 0 * -inf."""

    features, masks = _real_batch(env, encoder, 6)
    logits, _ = net(features, masks)
    distribution = Categorical(logits=logits)

    samples = distribution.sample()
    assert torch.isfinite(distribution.log_prob(samples)).all()
    assert torch.isfinite(distribution.entropy()).all()
    assert torch.all(distribution.entropy() >= 0.0)


def test_masks_that_cannot_be_satisfied_are_rejected(net, env, encoder):
    features, masks = _real_batch(env, encoder, 2)

    with pytest.raises(ValueError):
        net(features, torch.zeros(2, NUM_ACTIONS, dtype=torch.bool))
    with pytest.raises(ValueError):
        net(features, torch.ones(2, NUM_ACTIONS))
    with pytest.raises(ValueError):
        net(features, torch.ones(1, NUM_ACTIONS, dtype=torch.bool))
    with pytest.raises(ValueError):
        net(features, torch.ones(2, NUM_ACTIONS - 1, dtype=torch.bool))


def test_masked_logits_receive_no_gradient_but_the_trunk_does(net, env, encoder):
    """No signal at an illegal action -- not a negative signal."""

    features, masks = _real_batch(env, encoder, 1)
    assert not bool(masks.all())

    logits, value = net(features, masks)
    logits.retain_grad()

    distribution = Categorical(logits=logits)
    action = distribution.sample()
    loss = (
        -distribution.log_prob(action).mean()
        - 0.01 * distribution.entropy().mean()
        + 0.5 * value.pow(2).mean()
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.equal(
        logits.grad[~masks], torch.zeros_like(logits.grad[~masks])
    )
    assert logits.grad[masks].abs().sum() > 0
    assert net.trunk[0].weight.grad.abs().sum() > 0


# -- value head -----------------------------------------------------------
def test_value_head_ignores_the_legal_mask(net, env, encoder):
    features, _ = _real_batch(env, encoder, 1)

    _, wide = net(features, torch.ones(1, NUM_ACTIONS, dtype=torch.bool))
    _, narrow = net(
        features, torch.tensor([[True, False, True, False]])
    )

    assert torch.equal(wide, narrow)


# -- persistence and determinism ------------------------------------------
def test_network_is_deterministic_and_has_no_train_eval_difference(net, env, encoder):
    features, masks = _real_batch(env, encoder, 4)

    first_logits, first_value = net(features, masks)
    second_logits, second_value = net(features, masks)
    net.eval()
    eval_logits, eval_value = net(features, masks)

    assert torch.equal(first_logits, second_logits)
    assert torch.equal(first_value, second_value)
    assert torch.equal(first_logits, eval_logits)
    assert torch.equal(first_value, eval_value)


def test_state_dict_round_trip_reproduces_outputs(net, env, encoder):
    features, masks = _real_batch(env, encoder, 4)
    expected_logits, expected_value = net(features, masks)

    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)

    restored = PolicyValueNet()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    logits, value = restored(features, masks)

    assert torch.equal(logits, expected_logits)
    assert torch.equal(value, expected_value)


# -- against the real game ------------------------------------------------
def test_sampled_actions_are_always_legal_in_real_hands(net, env, encoder):
    """End to end: encode -> forward -> sample -> the action must be playable."""

    rng = np.random.default_rng(123)
    decisions = 0
    illegal = 0

    for _ in range(200):
        state, player_id = env.reset()
        small_blind = small_blind_player(state)

        while not env.is_over():
            encoded = encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            features = torch.from_numpy(encoded.features).unsqueeze(0)
            masks = torch.from_numpy(encoded.legal_mask).unsqueeze(0)

            with torch.no_grad():
                logits, value = net(features, masks)
                action = int(Categorical(logits=logits).sample())

            decisions += 1
            if action not in state["legal_actions"]:
                illegal += 1
            assert value.shape == (1,)

            state, player_id = env.step(action)

    assert decisions > 100
    assert illegal == 0
