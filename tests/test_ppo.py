"""Tests for the PPO update."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder
from poker_arena.environment import make_environment
from poker_arena.networks import NUM_ACTIONS, PolicyValueNet
from poker_arena.ppo import PPOConfig, PPOUpdater
from poker_arena.rollout import RolloutBatch, RolloutBuffer, collect_trajectories


def _new_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet()


def _clone(net: PolicyValueNet) -> PolicyValueNet:
    clone = PolicyValueNet()
    clone.load_state_dict(copy.deepcopy(net.state_dict()))
    return clone


def _synthetic_batch(
    net: PolicyValueNet,
    count: int = 16,
    *,
    advantages: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    return_offset: float = 0.0,
    log_prob_shift: float = 0.0,
    seed: int = 0,
) -> RolloutBatch:
    """A batch whose stored log-probs come from ``net`` unless shifted."""

    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, FEATURE_DIM, generator=generator)
    legal_masks = torch.ones(count, NUM_ACTIONS, dtype=torch.bool)
    actions = torch.randint(0, NUM_ACTIONS, (count,), generator=generator)

    with torch.no_grad():
        logits, values = net(features, legal_masks)
        log_probs = Categorical(logits=logits).log_prob(actions)

    if advantages is None:
        advantages = np.linspace(-1.0, 1.0, count)
    advantages = np.asarray(advantages, dtype=np.float32)
    if returns is None:
        returns = values.numpy().astype(np.float32) + advantages + return_offset
    returns = np.asarray(returns, dtype=np.float32)

    return RolloutBatch(
        features=features.numpy().astype(np.float32),
        legal_masks=legal_masks.numpy(),
        actions=actions.numpy().astype(np.int64),
        log_probs=(log_probs.numpy() + log_prob_shift).astype(np.float32),
        values=values.numpy().astype(np.float32),
        advantages=advantages,
        returns=returns,
    )


@pytest.fixture()
def net():
    return _new_net()


# -- configuration ---------------------------------------------------------
def test_default_config_is_pinned():
    config = PPOConfig()

    assert config.clip_epsilon == 0.2
    assert config.entropy_coef == 0.01
    assert config.value_coef == 0.5
    assert config.epochs == 4
    assert config.minibatch_size == 256
    assert config.max_grad_norm == 0.5
    assert config.learning_rate == 3e-4
    assert config.normalize_advantages is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"clip_epsilon": 0.0},
        {"clip_epsilon": 1.0},
        {"entropy_coef": -0.1},
        {"value_coef": -0.1},
        {"epochs": 0},
        {"minibatch_size": 0},
        {"max_grad_norm": 0.0},
        {"learning_rate": 0.0},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        PPOConfig(**overrides)


# -- one update ------------------------------------------------------------
def test_step_changes_parameters_and_reports_finite_diagnostics(net):
    batch = _synthetic_batch(net)
    before = copy.deepcopy(net.state_dict())

    stats = PPOUpdater(net, seed=0).step(batch)

    changed = [
        not torch.equal(before[name], parameter)
        for name, parameter in net.state_dict().items()
    ]
    assert any(changed)
    assert stats["epochs"] == 4
    assert stats["samples"] == len(batch)
    assert stats["minibatches"] == 4
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
    ):
        assert key in stats
        assert np.isfinite(stats[key]), key


def test_fresh_log_probs_start_inside_the_trust_region(net):
    """The first gradient step scores the sampling policy: ratio == 1 exactly."""

    batch = _synthetic_batch(net)
    config = PPOConfig(epochs=1, minibatch_size=len(batch))

    stats = PPOUpdater(net, config, seed=0).step(batch)

    assert stats["clip_fraction"] == 0.0
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-7)


def test_stale_log_probs_make_the_clip_bind(net):
    """Old log-probs are constants: a large gap means every sample is clipped."""

    batch = _synthetic_batch(net, log_prob_shift=5.0)  # ratio ~ e^-5
    config = PPOConfig(epochs=1, minibatch_size=len(batch))

    stats = PPOUpdater(net, config, seed=0).step(batch)

    assert stats["clip_fraction"] == 1.0
    assert stats["approx_kl"] > 0.0


def test_zero_advantages_leave_the_policy_head_alone_but_the_value_head_learns(net):
    """Advantages are the only path from the return to the policy."""

    batch = _synthetic_batch(net, advantages=np.zeros(16), return_offset=2.0)
    policy_before = copy.deepcopy(net.policy_head.state_dict())
    value_before = copy.deepcopy(net.value_head.state_dict())
    config = PPOConfig(
        entropy_coef=0.0, value_coef=0.5, epochs=1, minibatch_size=len(batch)
    )

    PPOUpdater(net, config, seed=0).step(batch)

    for name, parameter in net.policy_head.state_dict().items():
        assert torch.equal(parameter, policy_before[name]), name
    assert any(
        not torch.equal(parameter, value_before[name])
        for name, parameter in net.value_head.state_dict().items()
    )


def test_value_loss_pulls_the_value_towards_the_returns(net):
    batch = _synthetic_batch(
        net, advantages=np.zeros(16), return_offset=2.0
    )
    target = batch.returns.copy()

    with torch.no_grad():
        before = net(
            torch.as_tensor(batch.features),
            torch.as_tensor(batch.legal_masks),
        )[1].numpy()

    config = PPOConfig(
        entropy_coef=0.0,
        value_coef=1.0,
        epochs=40,
        minibatch_size=len(batch),
        learning_rate=1e-2,
    )
    PPOUpdater(net, config, seed=0).step(batch)

    with torch.no_grad():
        after = net(
            torch.as_tensor(batch.features),
            torch.as_tensor(batch.legal_masks),
        )[1].numpy()

    assert np.abs(after - target).mean() < np.abs(before - target).mean()


def test_entropy_bonus_increases_entropy(net):
    batch = _synthetic_batch(net, advantages=np.zeros(16))

    # The default policy head gain (0.01) leaves the policy almost uniform, so
    # there is no room for entropy to rise.  Sharpen it first.
    sharp = _clone(net)
    with torch.no_grad():
        sharp.policy_head.weight.mul_(20.0)
        sharp.policy_head.bias.mul_(20.0)

    plain = _clone(sharp)
    encouraged = _clone(sharp)

    with torch.no_grad():
        logits, _ = plain(
            torch.as_tensor(batch.features), torch.as_tensor(batch.legal_masks)
        )
        entropy_before = Categorical(logits=logits).entropy().mean().item()

    PPOUpdater(
        plain,
        PPOConfig(
            entropy_coef=0.0, value_coef=0.0, epochs=1,
            minibatch_size=len(batch), learning_rate=1e-3,
        ),
        seed=0,
    ).step(batch)
    PPOUpdater(
        encouraged,
        PPOConfig(
            entropy_coef=1.0, value_coef=0.0, epochs=1,
            minibatch_size=len(batch), learning_rate=1e-3,
        ),
        seed=0,
    ).step(batch)

    with torch.no_grad():
        plain_entropy = Categorical(
            logits=plain(
                torch.as_tensor(batch.features), torch.as_tensor(batch.legal_masks)
            )[0]
        ).entropy().mean().item()
        encouraged_entropy = Categorical(
            logits=encouraged(
                torch.as_tensor(batch.features), torch.as_tensor(batch.legal_masks)
            )[0]
        ).entropy().mean().item()

    assert encouraged_entropy > entropy_before
    assert plain_entropy == pytest.approx(entropy_before, abs=1e-7)


def test_advantage_normalisation_makes_the_update_scale_free(net):
    """Scaling every advantage by 1000 must not change the step."""

    scaled = _clone(net)
    original = _synthetic_batch(net)
    amplified = replace(
        original, advantages=(original.advantages * 1000.0).astype(np.float32)
    )

    config = PPOConfig(epochs=2, minibatch_size=len(original))
    PPOUpdater(net, config, seed=0).step(original)
    PPOUpdater(scaled, config, seed=0).step(amplified)

    for left, right in zip(net.parameters(), scaled.parameters()):
        assert torch.allclose(left, right, atol=1e-6)


def test_without_normalisation_a_huge_advantage_triggers_gradient_clipping(net):
    batch = _synthetic_batch(net, advantages=np.full(16, 1e4))
    config = PPOConfig(
        epochs=1, minibatch_size=len(batch), max_grad_norm=0.1,
        normalize_advantages=False,
    )

    stats = PPOUpdater(net, config, seed=0).step(batch)

    assert stats["grad_norm"] > config.max_grad_norm
    assert all(torch.isfinite(parameter).all() for parameter in net.parameters())


def test_minibatch_accounting(net):
    batch = _synthetic_batch(net, count=10)
    config = PPOConfig(epochs=2, minibatch_size=4)

    stats = PPOUpdater(net, config, seed=0).step(batch)

    assert stats["epochs"] == 2
    assert stats["samples"] == 10
    assert stats["minibatches"] == 6  # 3 + 3, the last one short


def test_seeded_updater_is_reproducible(net):
    batch = _synthetic_batch(net)
    twin = _clone(net)
    config = PPOConfig(epochs=2, minibatch_size=8)

    PPOUpdater(net, config, seed=7).step(batch)
    PPOUpdater(twin, config, seed=7).step(batch)

    for left, right in zip(net.parameters(), twin.parameters()):
        assert torch.equal(left, right)


def test_step_leaves_the_network_in_training_mode(net):
    net.eval()
    PPOUpdater(net, seed=0).step(_synthetic_batch(net))
    assert net.training


# -- validation ------------------------------------------------------------
def test_batch_validation(net):
    batch = _synthetic_batch(net, count=4)
    updater = PPOUpdater(net)

    with pytest.raises(ValueError):
        updater.step(replace(batch, returns=batch.returns[:-1]))
    with pytest.raises(ValueError):
        updater.step(replace(batch, log_probs=batch.log_probs[:-1]))
    with pytest.raises(ValueError):
        updater.step(replace(batch, features=batch.features[:, :-1]))
    with pytest.raises(ValueError):
        updater.step(replace(batch, legal_masks=batch.legal_masks[:, :-1]))
    with pytest.raises(ValueError):
        updater.step(
            replace(batch, legal_masks=batch.legal_masks.astype(np.float32))
        )
    with pytest.raises(ValueError):
        updater.step(
            replace(batch, advantages=np.full(4, np.nan, dtype=np.float32))
        )
    with pytest.raises(ValueError):
        updater.step(
            RolloutBatch(
                features=np.zeros((0, FEATURE_DIM), dtype=np.float32),
                legal_masks=np.zeros((0, NUM_ACTIONS), dtype=bool),
                actions=np.zeros(0, dtype=np.int64),
                log_probs=np.zeros(0, dtype=np.float32),
                values=np.zeros(0, dtype=np.float32),
                advantages=np.zeros(0, dtype=np.float32),
                returns=np.zeros(0, dtype=np.float32),
            )
        )


# -- against the real game -------------------------------------------------
def test_illegal_actions_stay_at_zero_probability_after_updates(net):
    env = make_environment(seed=21)
    encoder = ObservationEncoder.from_environment(env)
    trajectories = collect_trajectories(
        env, encoder, net, hands=20, rng=np.random.default_rng(2)
    )

    buffer = RolloutBuffer()
    for trajectory in trajectories:
        buffer.add_trajectory(trajectory)
    batch = buffer.batch()

    PPOUpdater(net, PPOConfig(epochs=2), seed=0).step(batch)

    with torch.no_grad():
        logits, _ = net(
            torch.as_tensor(batch.features),
            torch.as_tensor(batch.legal_masks),
        )
        probabilities = torch.softmax(logits, dim=-1)

    assert torch.all(probabilities[~torch.as_tensor(batch.legal_masks)] == 0.0)
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(len(batch))
    )


def test_rollout_update_rollout_cycle(net):
    env = make_environment(seed=33)
    encoder = ObservationEncoder.from_environment(env)

    first = collect_trajectories(
        env, encoder, net, hands=10, rng=np.random.default_rng(1)
    )
    buffer = RolloutBuffer()
    for trajectory in first:
        buffer.add_trajectory(trajectory)
    stats = PPOUpdater(net, seed=0).step(buffer.batch())
    assert np.isfinite(stats["total_loss"])

    second = collect_trajectories(
        env, encoder, net, hands=10, rng=np.random.default_rng(2)
    )
    assert second
    for trajectory in second:
        for transition in trajectory.transitions:
            assert bool(transition.legal_mask[transition.action])
