"""Tests for Agent B: the history-aware network, rollout, PPO and evaluation.

The important invariants are architectural: padding must not change the GRU state,
an empty history must be a true zero state, the ablation must cut the gradient to
the recurrent weights, and the training/evaluation protocol must match Agent A's
so the A/B comparison is about the model rather than the harness.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from poker_arena.encoding import FEATURE_DIM, ObservationEncoder, small_blind_player
from poker_arena.environment import make_environment
from poker_arena.history import MAX_HISTORY_TOKENS, TOKEN_DIM, HistoryTracker
from poker_arena.networks import MASKED_LOGIT, NUM_ACTIONS, PolicyValueNet
from poker_arena.networks_b import HISTORY_HIDDEN, HistoryPolicyValueNet
from poker_arena.rollout import compute_gae
from poker_arena.rollout_b import (
    HistoryRolloutBuffer,
    HistoryTrajectory,
    HistoryTransition,
    collect_history_trajectories,
)
from poker_arena.training_b import (
    AgentAPolicy,
    AgentBPolicy,
    TrainingConfigB,
    evaluate_paired_policies,
    evaluate_policy,
    load_checkpoint_b,
    save_checkpoint_b,
    train_b,
)

EXPECTED_B_PARAMETERS = 116_933
EXPECTED_A_PARAMETERS = 100_869


# -- fixtures and helpers ---------------------------------------------------
@pytest.fixture()
def env():
    return make_environment(seed=42)


@pytest.fixture()
def encoder(env):
    return ObservationEncoder.from_environment(env)


def _net(seed: int = 0) -> HistoryPolicyValueNet:
    torch.manual_seed(seed)
    return HistoryPolicyValueNet()


def _empty_history(count: int):
    """Torch tensors, because the network's public API takes tensors."""

    tokens = torch.zeros(count, MAX_HISTORY_TOKENS, TOKEN_DIM, dtype=torch.float32)
    mask = torch.zeros(count, MAX_HISTORY_TOKENS, dtype=torch.bool)
    return tokens, mask


def _real_batch(env, encoder, count: int = 8):
    """Encoded features, token tables, masks and legal masks from real hands."""

    rng = np.random.default_rng(0)
    features, tokens, histories, legal = [], [], [], []
    while len(features) < count:
        state, player_id = env.reset()
        small_blind = small_blind_player(state)
        tracker = HistoryTracker()
        tracker.start_hand(state)
        while not env.is_over() and len(features) < count:
            sequence = tracker.tokens(viewer=player_id)
            encoded = encoder.encode(
                state, is_small_blind=(player_id == small_blind)
            )
            features.append(encoded.features)
            tokens.append(sequence.tokens)
            histories.append(sequence.mask)
            legal.append(encoded.legal_mask)
            action = int(rng.choice(sorted(state["legal_actions"])))
            tracker.record(state, player_id, env.actions[action])
            state, player_id = env.step(action)
    return (
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(tokens)),
        torch.from_numpy(np.stack(histories)),
        torch.from_numpy(np.stack(legal)),
    )


def _transition(action: int = 0, value: float = 0.0) -> HistoryTransition:
    legal = np.zeros(NUM_ACTIONS, dtype=bool)
    legal[action] = True
    tokens, history = _empty_history(1)
    return HistoryTransition(
        features=np.zeros(FEATURE_DIM, dtype=np.float32),
        tokens=tokens[0].numpy(),
        history_mask=history[0].numpy(),
        legal_mask=legal,
        action=action,
        log_prob=0.0,
        value=value,
    )


def _tiny_config(tmp_path, **overrides) -> TrainingConfigB:
    settings = dict(
        iterations=2,
        hands_per_iteration=6,
        evaluation_every=1,
        evaluation_hands=6,
        seed=3,
        evaluation_seed=777,
        run_name="test_b",
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    settings.update(overrides)
    return TrainingConfigB(**settings)


# -- architecture -----------------------------------------------------------
def test_parameter_count_is_pinned_and_reported():
    net = _net()
    agent_a = PolicyValueNet()

    assert net.count_parameters() == EXPECTED_B_PARAMETERS
    assert agent_a.count_parameters() == EXPECTED_A_PARAMETERS
    # The extra capacity is the memory channel plus four wider head rows, not a
    # second trunk: B must not be "A but twice as big".
    assert net.count_parameters() - agent_a.count_parameters() == 16_064
    assert net.count_parameters() / agent_a.count_parameters() < 1.2
    assert net.hidden == agent_a.hidden
    assert net.history_hidden == HISTORY_HIDDEN


def test_forward_shapes_and_legal_mask(env, encoder):
    net = _net()
    features, tokens, history, legal = _real_batch(env, encoder)

    logits, value = net(features, tokens, history, legal)
    probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (len(features), NUM_ACTIONS)
    assert value.shape == (len(features),)
    assert torch.all(probabilities[~legal] == 0.0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(features)))
    assert torch.all(logits[~legal] == MASKED_LOGIT)
    assert torch.isfinite(value).all()


def test_empty_history_matches_history_ablation(env, encoder):
    net = _net()
    features, _, _, legal = _real_batch(env, encoder)
    tokens, history = _empty_history(len(features))

    with torch.no_grad():
        empty_logits, empty_value = net(features, tokens, history, legal)
        off_logits, off_value = net(
            features, tokens, history, legal, use_history=False
        )

    assert torch.equal(empty_logits, off_logits)
    assert torch.equal(empty_value, off_value)


def test_padding_does_not_change_the_gru_state():
    net = _net()
    tokens = torch.zeros(1, MAX_HISTORY_TOKENS, TOKEN_DIM)
    tokens[0, 0, 3] = 1.0
    tokens[0, 1, 7] = 1.0
    mask = torch.zeros(1, MAX_HISTORY_TOKENS, dtype=torch.bool)
    mask[0, :2] = True

    with torch.no_grad():
        packed = net.history(tokens, mask)
        manual, _ = net.history.gru(tokens[0, :2].unsqueeze(0))  # no padding
        naive = net.history.gru(tokens)[0][:, -1]  # recurrent over all 32 rows

    # The packed state must equal feeding only the two real tokens.
    np.testing.assert_allclose(
        packed.numpy(), manual[:, -1].numpy(), rtol=0, atol=1e-6
    )
    # A GRU updates on a zero vector through its biases, so the naive run over
    # the padded table must *not* agree.  This is why packing is required.
    assert not torch.allclose(packed[0], naive[0], atol=1e-5)


def test_padding_count_changes_nothing():
    net = _net()
    tokens = torch.zeros(2, MAX_HISTORY_TOKENS, TOKEN_DIM)
    tokens[:, 0, 1] = 1.0
    mask = torch.zeros(2, MAX_HISTORY_TOKENS, dtype=torch.bool)
    mask[:, 0] = True
    tokens[1, 1:, :] = 0.0

    with torch.no_grad():
        first = net.history(tokens, mask)

    # The two rows carry the same single token, so their states must match.
    np.testing.assert_allclose(
        first[0].numpy(), first[1].numpy(), rtol=0, atol=0
    )


def test_hole_in_history_mask_is_rejected(env, encoder):
    net = _net()
    features, tokens, history, legal = _real_batch(env, encoder)
    broken = history.clone()
    broken[0, 0] = True
    broken[0, 1] = False
    broken[0, 2] = True

    with pytest.raises(ValueError, match="prefix"):
        net(features, tokens, broken, legal)


def test_wrong_history_shape_is_rejected(env, encoder):
    net = _net()
    features, tokens, history, legal = _real_batch(env, encoder)

    with pytest.raises(ValueError, match="token_dim|tokens must have width"):
        net(features, tokens[:, :, :-1], history, legal)
    with pytest.raises(ValueError, match="history mask"):
        net(features, tokens, history[:, :-1], legal)
    with pytest.raises(ValueError, match="does not match"):
        net(features, tokens[:-1], history[:-1], legal)


def test_batch_rows_are_independent(env, encoder):
    net = _net()
    features, tokens, history, legal = _real_batch(env, encoder)

    with torch.no_grad():
        batch_logits, _ = net(features, tokens, history, legal)
        single_logits, _ = net(
            features[3:4], tokens[3:4], history[3:4], legal[3:4]
        )

    assert torch.allclose(batch_logits[3], single_logits[0], atol=1e-6)


def test_use_history_flag_controls_the_recurrent_gradient(env, encoder):
    features, tokens, history, legal = _real_batch(env, encoder)

    net = _net()
    net.zero_grad()
    logits, value = net(features, tokens, history, legal)
    (logits.sum() + value.sum()).backward()
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in net.history.gru.parameters()
    )

    control = _net()
    control.zero_grad()
    logits, value = control(features, tokens, history, legal, use_history=False)
    (logits.sum() + value.sum()).backward()
    assert all(
        parameter.grad is None
        for parameter in control.history.gru.parameters()
    )
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in control.trunk.parameters()
    )


def test_history_changes_the_policy():
    net = _net()
    features = torch.zeros(1, FEATURE_DIM)
    legal = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    tokens, history = _empty_history(1)
    tokens[0, 0] = torch.arange(TOKEN_DIM, dtype=torch.float32) / 16.0
    history[0, 0] = True

    with torch.no_grad():
        with_logits, _ = net(features, tokens, history, legal)
        without_logits, _ = net(
            features, tokens, history, legal, use_history=False
        )

    assert not torch.allclose(with_logits, without_logits)
    assert (with_logits - without_logits).abs().max() > 1e-6


# -- rollout ----------------------------------------------------------------
def test_collected_histories_are_prefixes_of_increasing_length(env, encoder):
    net = _net()
    trajectories = collect_history_trajectories(
        env, encoder, net, hands=3, rng=np.random.default_rng(0)
    )

    assert trajectories
    for trajectory in trajectories:
        lengths = [int(t.history_mask.sum()) for t in trajectory.transitions]
        assert lengths == sorted(set(lengths)), "each decision adds at least one action"
        assert all(0 <= length <= MAX_HISTORY_TOKENS for length in lengths)
        for transition in trajectory.transitions:
            length = int(transition.history_mask.sum())
            assert transition.history_mask[:length].all()
            assert not transition.history_mask[length:].any()
            assert np.all(transition.tokens[length:] == 0.0)
            assert transition.legal_mask[transition.action]
            assert np.isfinite(transition.log_prob)
            assert np.isfinite(transition.value)


def test_self_play_records_both_seats_and_opponent_mode_records_one(env, encoder):
    net = _net()

    self_play = collect_history_trajectories(
        env, encoder, net, hands=4, rng=np.random.default_rng(1)
    )
    against = collect_history_trajectories(
        env,
        encoder,
        net,
        hands=4,
        rng=np.random.default_rng(1),
        opponent=lambda state, player_id: 0 if 0 in state["legal_actions"] else 3,
    )

    # Self-play records both seats, so four hands produce between one and two
    # trajectories each.  Opponent mode records only player 0, so at most one
    # trajectory per hand and sometimes none.
    assert 4 <= len(self_play) <= 8
    assert len(against) <= 4
    assert against


def test_collection_restores_training_mode(env, encoder):
    net = _net()
    net.train()
    collect_history_trajectories(
        env, encoder, net, hands=1, rng=np.random.default_rng(0)
    )
    assert net.training


def test_collection_is_reproducible(env, encoder):
    features, tokens, histories = [], [], []
    for _ in range(2):
        net = _net()
        # A collection consumes the environment's deal stream, so rewind it:
        # reproducibility is about the same deals and the same action samples.
        env.seed(42)
        trajectories = collect_history_trajectories(
            env, encoder, net, hands=2, rng=np.random.default_rng(5)
        )
        features.append([t.features for trajectory in trajectories
                         for t in trajectory.transitions])
        tokens.append([t.tokens for trajectory in trajectories
                       for t in trajectory.transitions])
        histories.append([t.history_mask for trajectory in trajectories
                          for t in trajectory.transitions])

    assert len(features[0]) == len(features[1])
    for first, second in zip(features[0], features[1]):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(tokens[0], tokens[1]):
        np.testing.assert_array_equal(first, second)


def test_buffer_rejects_a_non_prefix_mask():
    tokens, history = _empty_history(1)
    history[0, 2] = True
    broken = HistoryTransition(
        features=np.zeros(FEATURE_DIM, dtype=np.float32),
        tokens=tokens[0].numpy(),
        history_mask=history[0].numpy(),
        legal_mask=np.array([True, False, False, False]),
        action=0,
        log_prob=0.0,
        value=0.0,
    )

    with pytest.raises(ValueError, match="prefix"):
        HistoryRolloutBuffer().add_trajectory(
            HistoryTrajectory((broken,), final_payoff=0.0)
        )


def test_buffer_rejects_bad_shapes_and_illegal_actions():
    tokens, history = _empty_history(1)
    good = _transition()

    bad_tokens = HistoryTransition(
        features=good.features,
        tokens=tokens[0][:, :-1],
        history_mask=good.history_mask,
        legal_mask=good.legal_mask,
        action=0,
        log_prob=0.0,
        value=0.0,
    )
    with pytest.raises(ValueError, match="tokens must be"):
        HistoryRolloutBuffer().add_trajectory(
            HistoryTrajectory((bad_tokens,), final_payoff=0.0)
        )

    illegal = HistoryTransition(
        features=good.features,
        tokens=good.tokens,
        history_mask=good.history_mask,
        legal_mask=np.array([False, True, False, False]),
        action=0,
        log_prob=0.0,
        value=0.0,
    )
    with pytest.raises(ValueError, match="illegal"):
        HistoryRolloutBuffer().add_trajectory(
            HistoryTrajectory((illegal,), final_payoff=0.0)
        )


def test_buffer_batch_matches_compute_gae():
    trajectory = HistoryTrajectory(
        transitions=(_transition(0, 0.5), _transition(3, 1.0)),
        final_payoff=2.0,
    )
    buffer = HistoryRolloutBuffer(gamma=1.0, lam=0.95)
    buffer.add_trajectory(trajectory)
    batch = buffer.batch()

    advantages, returns = compute_gae(
        trajectory.rewards(), trajectory.values(), gamma=1.0, lam=0.95
    )
    np.testing.assert_allclose(batch.advantages, advantages, rtol=1e-6)
    np.testing.assert_allclose(batch.returns, returns, rtol=1e-6)
    assert len(batch) == 2
    assert batch.tokens.shape == (2, MAX_HISTORY_TOKENS, TOKEN_DIM)
    assert batch.history_masks.shape == (2, MAX_HISTORY_TOKENS)


# -- evaluation -------------------------------------------------------------
def test_evaluate_policy_reports_the_same_statistics(env, encoder):
    net = _net()
    policy = AgentBPolicy(net, encoder)
    result = evaluate_policy(
        policy, 40, seed=123, opponent_name="raiser_strong"
    )

    assert result.hands == 40
    assert result.opponent == "raiser_strong"
    assert np.isfinite(result.mean_bb)
    assert result.ci95_bb == pytest.approx(1.959963984540054 * result.stderr_bb)
    assert result.stderr_bb > 0


def test_paired_evaluation_of_one_policy_is_exactly_zero(env, encoder):
    net = _net()
    policy = AgentBPolicy(net, encoder)

    result = evaluate_paired_policies(
        policy, policy, 30, seed=456, opponent_name="raiser_loose"
    )

    assert result.mean_difference_bb == 0.0
    assert result.stderr_difference_bb == 0.0
    assert result.ci95_difference_bb == 0.0
    assert result.wins == 0 and result.losses == 0 and result.ties == 30
    assert all(value == 0.0 for value in result.differences_bb)


def test_paired_ablation_uses_the_same_weights(env, encoder):
    net = _net()
    with_history = AgentBPolicy(net, encoder, use_history=True)
    without = AgentBPolicy(net, encoder, use_history=False)

    result = evaluate_paired_policies(
        with_history, without, 40, seed=99, opponent_name="passive"
    )

    assert result.hands == 40
    assert result.wins + result.losses + result.ties == 40
    assert result.total_difference_bb == pytest.approx(sum(result.differences_bb))
    assert result.ci95_difference_bb == pytest.approx(
        1.959963984540054 * result.stderr_difference_bb
    )


def test_paired_evaluation_is_reproducible(env, encoder):
    net_a = _net(0)
    net_b = _net(1)
    policy_a = AgentBPolicy(net_a, encoder)
    policy_b = AgentBPolicy(net_b, encoder)

    first = evaluate_paired_policies(
        policy_a, policy_b, 25, seed=321, opponent_name="raiser_strong"
    )
    second = evaluate_paired_policies(
        policy_a, policy_b, 25, seed=321, opponent_name="raiser_strong"
    )

    assert first.differences_bb == second.differences_bb
    assert first.mean_difference_bb == second.mean_difference_bb


def test_agent_a_policy_ignores_the_history_tracker(env, encoder):
    agent_a = AgentAPolicy(PolicyValueNet(), encoder)
    tracker = HistoryTracker()
    state, player_id = env.reset()
    tracker.start_hand(state)
    first = agent_a(state, player_id, tracker, small_blind_player(state))

    tracker.record(state, player_id, "call")
    second = agent_a(state, player_id, tracker, small_blind_player(state))

    assert first == second


def test_a_versus_b_paired_result_counters_add_up(env, encoder):
    agent_a = AgentAPolicy(PolicyValueNet(), encoder)
    agent_b = AgentBPolicy(_net(), encoder)

    result = evaluate_paired_policies(
        agent_a, agent_b, 30, seed=2024, opponent_name="raiser_loose"
    )

    assert result.wins + result.losses + result.ties == 30
    assert len(result.differences_bb) == 30


# -- checkpoints and training ----------------------------------------------
def test_checkpoint_round_trip(tmp_path, encoder):
    net = _net()
    config = _tiny_config(tmp_path)
    path = tmp_path / "agent_b.pt"

    saved = save_checkpoint_b(net, config, 7, path)
    loaded, metadata = load_checkpoint_b(saved)

    assert metadata["iteration"] == 7
    assert metadata["config"]["run_name"] == "test_b"
    assert loaded.history_hidden == net.history_hidden
    for name, parameter in net.state_dict().items():
        assert torch.equal(parameter, loaded.state_dict()[name])


def test_training_config_b_validates_history_width(tmp_path):
    with pytest.raises(ValueError, match="history_hidden"):
        _tiny_config(tmp_path, history_hidden=0)


def test_train_b_writes_logs_checkpoints_and_changes_weights(tmp_path, encoder):
    config = _tiny_config(tmp_path)
    net = _net()
    before = {name: value.clone() for name, value in net.state_dict().items()}

    result = train_b(config, net=net)

    assert len(result.iterations) == 2
    assert len(result.evaluations) == 3 * 3  # three checkpoints, three opponents
    assert len(result.checkpoints) == 3
    assert all("decisions" in row for row in result.iterations)
    assert all(np.isfinite(row["policy_loss"]) for row in result.iterations)
    assert any(
        not torch.equal(before[name], value)
        for name, value in net.state_dict().items()
    )

    with open(result.train_log, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    with open(result.evaluation_log, newline="", encoding="utf-8") as handle:
        evaluations = list(csv.DictReader(handle))
    assert len(evaluations) == 9
    assert {row["opponent"] for row in evaluations} == {
        "passive",
        "raiser_strong",
        "raiser_loose",
    }


def test_train_b_is_reproducible(tmp_path, encoder):
    config_a = _tiny_config(tmp_path, run_name="repro_a")
    config_b = _tiny_config(tmp_path, run_name="repro_b")
    net_a = _net()
    net_b = _net()

    train_b(config_a, net=net_a)
    train_b(config_b, net=net_b)

    for name, parameter in net_a.state_dict().items():
        assert torch.equal(parameter, net_b.state_dict()[name])


def test_train_b_control_arm_runs(tmp_path, encoder):
    config = _tiny_config(tmp_path, use_history=False, run_name="control")
    result = train_b(config, net=_net())

    assert len(result.iterations) == 2
    assert all(np.isfinite(row["value_loss"]) for row in result.iterations)
