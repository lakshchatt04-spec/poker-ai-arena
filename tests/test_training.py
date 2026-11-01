"""Tests for the training loop and fixed-seed evaluation."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
import torch

from poker_arena.networks import PolicyValueNet
from poker_arena.training import (
    Z_95,
    EvaluationRecord,
    EvaluationResult,
    TrainingConfig,
    evaluate,
    load_checkpoint,
    random_opponent,
    save_checkpoint,
    scripted_opponent,
    train,
)


def _new_net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet()


def _tiny_config(tmp_path, **overrides) -> TrainingConfig:
    settings = dict(
        iterations=2,
        hands_per_iteration=6,
        evaluation_every=1,
        evaluation_hands=12,
        seed=3,
        evaluation_seed=777,
        run_name="test_run",
        log_dir=str(tmp_path / "runs"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    settings.update(overrides)
    return TrainingConfig(**settings)


# -- configuration ---------------------------------------------------------
def test_default_config_is_pinned():
    config = TrainingConfig()

    assert config.iterations == 50
    assert config.hands_per_iteration == 320
    assert config.evaluation_every == 10
    assert config.evaluation_hands == 2_000
    assert config.seed == 0
    assert config.evaluation_seed == 10_000
    assert config.run_name == "agent_a"
    assert config.log_dir == "runs"
    assert config.checkpoint_dir == "checkpoints"
    assert config.ppo.learning_rate == 3e-4


@pytest.mark.parametrize(
    "overrides",
    [
        {"iterations": 0},
        {"hands_per_iteration": 0},
        {"evaluation_every": 0},
        {"evaluation_hands": 1},
        {"run_name": "   "},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)


# -- opponents -------------------------------------------------------------
def test_scripted_opponent_calls_then_checks_then_folds():
    assert scripted_opponent({"legal_actions": [0, 1, 2, 3]}, 1) == 0
    assert scripted_opponent({"legal_actions": [1, 2, 3]}, 1) == 3
    assert scripted_opponent({"legal_actions": [1, 2]}, 1) == 2


def test_random_opponent_is_seeded_and_always_legal():
    state = {"legal_actions": [1, 2, 3]}

    first = random_opponent(np.random.default_rng(5))
    second = random_opponent(np.random.default_rng(5))

    left = [first(state, 1) for _ in range(20)]
    right = [second(state, 1) for _ in range(20)]

    assert left == right
    assert set(left) <= set(state["legal_actions"])


# -- evaluation ------------------------------------------------------------
def test_evaluate_reports_finite_statistics_and_a_z_interval():
    net = _new_net()

    result = evaluate(net, 200, seed=123)

    assert isinstance(result, EvaluationResult)
    assert result.hands == 200
    assert result.opponent == "scripted"
    assert np.isfinite(result.mean_bb)
    assert np.isfinite(result.stderr_bb)
    assert result.stderr_bb >= 0.0
    assert result.ci95_bb == pytest.approx(Z_95 * result.stderr_bb)
    assert result.total_bb == pytest.approx(result.mean_bb * result.hands)
    # The game pays in whole and half big blinds.
    assert abs(result.total_bb * 2.0 - round(result.total_bb * 2.0)) < 1e-9


def test_evaluate_is_deterministic_for_the_same_weights_and_seed():
    net = _new_net()

    first = evaluate(net, 40, seed=99)
    second = evaluate(copy.deepcopy(net), 40, seed=99)

    assert first == second


def test_evaluate_restores_the_training_mode():
    net = _new_net()

    net.train()
    evaluate(net, 10, seed=1)
    assert net.training

    net.eval()
    evaluate(net, 10, seed=1)
    assert not net.training


def test_evaluate_accepts_a_random_opponent():
    opponent = random_opponent(np.random.default_rng(0))
    result = evaluate(
        _new_net(), 30, seed=2, opponent=opponent, opponent_name="random"
    )

    assert result.opponent == "random"
    assert np.isfinite(result.mean_bb)


def test_evaluate_rejects_fewer_than_two_hands():
    with pytest.raises(ValueError):
        evaluate(_new_net(), 1)


# -- checkpoints -----------------------------------------------------------
def test_checkpoint_round_trip_preserves_weights_and_metadata(tmp_path):
    net = _new_net(4)
    config = _tiny_config(tmp_path, iterations=7, evaluation_every=3)
    path = tmp_path / "nested" / "agent.pt"

    written = save_checkpoint(net, config, 42, path)
    restored, metadata = load_checkpoint(written)

    assert written == str(path)
    assert metadata["iteration"] == 42
    assert metadata["config"]["iterations"] == 7
    assert metadata["config"]["evaluation_every"] == 3
    for name, tensor in net.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


# -- the training loop -----------------------------------------------------
def test_train_writes_one_row_per_iteration_and_its_csvs(tmp_path):
    config = _tiny_config(tmp_path)
    net = _new_net()

    result = train(config, net=net)

    assert len(result.iterations) == config.iterations
    assert [row["iteration"] for row in result.iterations] == [1, 2]
    assert all(row["decisions"] > 0 for row in result.iterations)
    assert all(np.isfinite(row["total_loss"]) for row in result.iterations)

    with open(result.train_log, newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    assert len(train_rows) == 2
    assert set(train_rows[0]) == set(result.iterations[0]) | {"iteration"}
    assert int(train_rows[0]["iteration"]) == 1
    assert float(train_rows[0]["total_loss"]) == pytest.approx(
        float(result.iterations[0]["total_loss"])
    )

    with open(result.evaluation_log, newline="", encoding="utf-8") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    assert len(evaluation_rows) == 2
    assert all(row["opponent"] == "scripted" for row in evaluation_rows)


def test_train_evaluates_on_the_final_iteration_even_off_schedule(tmp_path):
    config = _tiny_config(tmp_path, iterations=3, evaluation_every=2)

    result = train(config)

    assert [record.iteration for record in result.evaluations] == [2, 3]
    assert len(result.checkpoints) == 2
    assert all(path.endswith(".pt") for path in result.checkpoints)
    assert all(
        isinstance(record, EvaluationRecord) for record in result.evaluations
    )


def test_train_changes_the_weights(tmp_path):
    net = _new_net()
    before = copy.deepcopy(net.state_dict())

    train(_tiny_config(tmp_path), net=net)

    changed = [
        not torch.equal(before[name], tensor)
        for name, tensor in net.state_dict().items()
    ]
    assert any(changed)


def test_train_reports_every_iteration_to_the_callback(tmp_path):
    seen: list[tuple[int, int | None]] = []

    def progress(row, record):
        seen.append((int(row["iteration"]), None if record is None else record.iteration))

    train(_tiny_config(tmp_path, iterations=3, evaluation_every=2), progress=progress)

    assert seen == [(1, None), (2, 2), (3, 3)]


def test_train_is_reproducible_from_the_same_config(tmp_path):
    first_config = _tiny_config(tmp_path / "a")
    second_config = _tiny_config(tmp_path / "b")

    first_net = _new_net(11)
    second_net = _new_net(11)
    first = train(first_config, net=first_net)
    second = train(second_config, net=second_net)

    assert first.iterations == second.iterations
    assert first.evaluations == second.evaluations
    for name, tensor in first_net.state_dict().items():
        assert torch.equal(tensor, second_net.state_dict()[name]), name


def test_evaluation_is_reproducible_from_a_saved_checkpoint(tmp_path):
    config = _tiny_config(tmp_path)
    net = _new_net()

    result = train(config, net=net)
    first = evaluate(net, 25, seed=555)

    restored, metadata = load_checkpoint(result.checkpoints[-1])
    second = evaluate(restored, 25, seed=555)

    assert metadata["iteration"] == config.iterations
    assert first == second
