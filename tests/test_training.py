"""Tests for the training loop and fixed-seed evaluation."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
import torch

from poker_arena.networks import PolicyValueNet
from poker_arena.training import (
    OPPONENT_NAMES,
    Z_95,
    EvaluationRecord,
    EvaluationResult,
    PairedEvaluationResult,
    TrainingConfig,
    evaluate,
    evaluate_paired,
    hole_strength,
    load_checkpoint,
    make_opponent,
    raising_opponent,
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
    assert config.self_play_fraction == 0.5
    assert config.training_opponent == "raiser_strong"
    assert config.evaluation_opponents == (
        "passive",
        "raiser_strong",
        "raiser_loose",
    )
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
        {"self_play_fraction": -0.1},
        {"self_play_fraction": 1.5},
        {"training_opponent": "nope"},
        {"evaluation_opponents": ()},
        {"evaluation_opponents": ("passive", "nope")},
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


def test_hole_strength_ranks_cards_without_a_full_evaluator():
    def strength(*cards: str) -> float:
        return hole_strength({"raw_obs": {"hand": list(cards)}})

    trash = strength("S2", "H7")
    suited = strength("SA", "SK")  # two big cards, same suit
    pair = strength("S7", "H7")

    assert 0.0 <= trash < suited
    assert pair > trash
    # Missing or malformed cards must not crash a scripted bot.
    assert hole_strength({}) == 0.0
    assert hole_strength({"raw_obs": {"hand": ["bad"]}}) == 0.0


def test_raising_opponent_raises_only_when_told_and_otherwise_checks():
    always = raising_opponent(lambda state: True)
    never = raising_opponent(lambda state: False)

    assert always({"legal_actions": [0, 1, 2]}, 1) == 1
    assert always({"legal_actions": [0, 2]}, 1) == 0  # no raise available: call
    assert never({"legal_actions": [1, 2, 3]}, 1) == 3  # check before fold
    assert never({"legal_actions": [1, 2]}, 1) == 2


def test_make_opponent_builds_the_named_style():
    strong = make_opponent("raiser_strong")
    weak_state = {"raw_obs": {"hand": ["S2", "H7"]}, "legal_actions": [0, 1, 2]}
    strong_state = {"raw_obs": {"hand": ["SA", "SK"]}, "legal_actions": [0, 1, 2]}

    assert make_opponent("passive") is scripted_opponent
    assert strong(weak_state, 1) == 0  # trash: call, do not raise
    assert strong(strong_state, 1) == 1  # premium: raise


def test_loose_raiser_is_reproducible_and_card_independent():
    first = make_opponent("raiser_loose", seed=4)
    second = make_opponent("raiser_loose", seed=4)

    state = {"raw_obs": {"hand": ["S2", "H7"]}, "legal_actions": [0, 1, 2]}
    left = [first(state, 1) for _ in range(2_000)]
    right = [second(state, 1) for _ in range(2_000)]

    assert left == right
    raise_rate = sum(action == 1 for action in left) / len(left)
    assert raise_rate == pytest.approx(0.35, abs=0.05)


def test_make_opponent_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown opponent"):
        make_opponent("nope")

    assert set(OPPONENT_NAMES) == {
        "passive",
        "raiser_strong",
        "raiser_loose",
        "random",
        "c_bettor",
        "first_raiser",
        "coin_opener",
    }


# -- evaluation ------------------------------------------------------------
def test_evaluate_reports_finite_statistics_and_a_z_interval():
    net = _new_net()

    result = evaluate(net, 200, seed=123)

    assert isinstance(result, EvaluationResult)
    assert result.hands == 200
    assert result.opponent == "passive"
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


# -- paired evaluation -----------------------------------------------------
def test_paired_evaluation_of_identical_networks_is_exactly_zero():
    net = _new_net()

    result = evaluate_paired(net, copy.deepcopy(net), 40, seed=321)

    assert isinstance(result, PairedEvaluationResult)
    assert result.hands == 40
    assert result.opponent == "passive"
    assert result.mean_difference_bb == 0.0
    assert result.stderr_difference_bb == 0.0
    assert result.ci95_difference_bb == 0.0
    assert result.total_difference_bb == 0.0
    assert result.ties == 40
    assert result.wins == 0
    assert result.losses == 0
    assert result.differences_bb == (0.0,) * 40
    # Different deals still have spread; it is the *difference* that is zero.
    assert result.unpaired_stderr_bb > 0.0


def test_paired_evaluation_also_replays_a_stochastic_opponent():
    # raiser_loose draws its raises from its own RNG.  Per-hand construction from
    # the deal seed makes that stream identical for both networks too, so two
    # copies of one network must agree on every hand.
    net = _new_net()

    result = evaluate_paired(
        net, copy.deepcopy(net), 30, seed=77, opponent_name="raiser_loose"
    )

    assert result.ties == 30
    assert result.mean_difference_bb == 0.0


def test_paired_evaluation_is_reproducible_and_tighter_than_independent_means():
    net_a = _new_net(0)
    net_b = _new_net(1)

    first = evaluate_paired(net_a, net_b, 300, seed=42)
    second = evaluate_paired(
        copy.deepcopy(net_a), copy.deepcopy(net_b), 300, seed=42
    )

    assert first == second
    assert first.hands == 300
    assert first.wins + first.losses + first.ties == 300
    assert first.mean_difference_bb == pytest.approx(
        first.mean_a_bb - first.mean_b_bb
    )
    assert first.ci95_difference_bb == pytest.approx(
        Z_95 * first.stderr_difference_bb
    )
    assert first.total_difference_bb == pytest.approx(
        first.mean_difference_bb * first.hands
    )
    # The networks play similar policies on the same deals, so the deal noise
    # cancels and the paired interval is tighter than adding two variances.
    assert first.stderr_difference_bb < first.unpaired_stderr_bb


def test_paired_evaluation_restores_both_training_modes():
    net_a = _new_net(0)
    net_b = _new_net(1)
    net_a.train()
    net_b.eval()

    evaluate_paired(net_a, net_b, 10, seed=5)

    assert net_a.training
    assert not net_b.training


def test_paired_evaluation_accepts_an_explicit_opponent():
    net = _new_net()

    result = evaluate_paired(
        net,
        copy.deepcopy(net),
        10,
        seed=1,
        opponent=scripted_opponent,
        opponent_name="custom",
    )

    assert result.opponent == "custom"
    assert result.ties == 10


def test_paired_evaluation_rejects_an_unknown_opponent():
    net = _new_net()

    with pytest.raises(ValueError):
        evaluate_paired(net, copy.deepcopy(net), 10, opponent_name="nope")


def test_paired_evaluation_rejects_fewer_than_two_hands():
    net = _new_net()

    with pytest.raises(ValueError):
        evaluate_paired(net, copy.deepcopy(net), 1)


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
    # iteration 0 plus the two scheduled iterations, one row per opponent
    assert len(evaluation_rows) == 9
    assert [int(row["iteration"]) for row in evaluation_rows] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2
    ]
    assert [row["opponent"] for row in evaluation_rows] == list(
        config.evaluation_opponents
    ) * 3


def test_train_evaluates_on_the_final_iteration_even_off_schedule(tmp_path):
    config = _tiny_config(tmp_path, iterations=3, evaluation_every=2)

    result = train(config)

    assert [record.iteration for record in result.evaluations] == [
        0, 0, 0, 2, 2, 2, 3, 3, 3
    ]
    assert len(result.checkpoints) == 3
    assert result.checkpoints[0].endswith("_0000.pt")
    assert all(path.endswith(".pt") for path in result.checkpoints)
    assert all(
        isinstance(record, EvaluationRecord) for record in result.evaluations
    )


def test_the_initial_evaluation_measures_the_starting_network(tmp_path):
    """Iteration 0 must play the untouched network, not a trained one."""

    net = _new_net()
    pristine = copy.deepcopy(net)

    result = train(_tiny_config(tmp_path, iterations=1), net=net)

    assert [record.iteration for record in result.evaluations] == [0, 0, 0, 1, 1, 1]

    expected = evaluate(pristine, hands=12, seed=777)
    assert result.evaluations[0].result == expected


def test_train_changes_the_weights(tmp_path):
    net = _new_net()
    before = copy.deepcopy(net.state_dict())

    train(_tiny_config(tmp_path), net=net)

    changed = [
        not torch.equal(before[name], tensor)
        for name, tensor in net.state_dict().items()
    ]
    assert any(changed)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_train_accepts_the_edges_of_the_mixture(tmp_path, fraction):
    config = _tiny_config(
        tmp_path,
        iterations=1,
        self_play_fraction=fraction,
        evaluation_opponents=("passive",),
    )

    result = train(config)

    assert len(result.iterations) == 1
    assert all(np.isfinite(row["total_loss"]) for row in result.iterations)
    assert [record.result.opponent for record in result.evaluations] == [
        "passive", "passive"
    ]


def test_train_reports_every_iteration_to_the_callback(tmp_path):
    seen: list[tuple[int, tuple[int, ...] | None]] = []

    def progress(row, records):
        seen.append(
            (
                int(row["iteration"]),
                None if records is None
                else tuple(record.iteration for record in records),
            )
        )

    train(_tiny_config(tmp_path, iterations=3, evaluation_every=2), progress=progress)

    assert seen == [(1, None), (2, (2, 2, 2)), (3, (3, 3, 3))]


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
