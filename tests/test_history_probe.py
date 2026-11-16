"""Tests for the within-summary card-leak statistic in ``history_probe``.

The probe's headline claim is that the hidden mode is absent from the public
summary. A global classifier can fake that by learning each summary's majority,
so the leak test has to compare scores inside one summary. These tests pin the
AUC and its permutation null.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import history_probe


def test_within_group_auc_perfect_separation():
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    keys = [b"a", b"a", b"b", b"b"]
    auc, pairs, groups, p_value = history_probe._within_group_auc(
        scores, labels, keys, permutations=50
    )
    assert auc == 1.0
    assert pairs == 2
    assert groups == 2
    assert p_value < 0.5


def test_within_group_auc_reversed():
    scores = np.array([1.0, 0.0])
    labels = np.array([0, 1])
    keys = [b"a", b"a"]
    auc, pairs, groups, _ = history_probe._within_group_auc(
        scores, labels, keys, permutations=10
    )
    assert auc == 0.0
    assert pairs == 1
    assert groups == 1


def test_within_group_auc_ties_are_half():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    keys = [b"a", b"a", b"b", b"b"]
    auc, _, _, _ = history_probe._within_group_auc(
        scores, labels, keys, permutations=10
    )
    assert auc == 0.5


def test_within_group_auc_ignores_single_mode_groups():
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    keys = [b"a", b"a", b"b", b"b"]
    auc, pairs, groups, _ = history_probe._within_group_auc(
        scores, labels, keys, permutations=10
    )
    assert np.isnan(auc)
    assert pairs == 0
    assert groups == 0


def test_within_group_auc_permutation_null_is_small_for_a_real_leak():
    scores = []
    labels = []
    keys = []
    for group in range(20):
        for index in range(5):
            keys.append(bytes([group]))
            scores.append(0.0)
            labels.append(0)
        for index in range(5):
            keys.append(bytes([group]))
            scores.append(1.0)
            labels.append(1)
    auc, pairs, groups, p_value = history_probe._within_group_auc(
        np.array(scores), np.array(labels), keys, permutations=100
    )
    assert auc == 1.0
    assert pairs == 20 * 25
    assert groups == 20
    assert p_value <= 0.02


def test_fit_logistic_matches_fit_scores_sign():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(200, 3))
    y_train = (x_train[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(float)
    x_test = rng.normal(size=(50, 3))
    y_test = (x_test[:, 0] > 0).astype(float)
    accuracy = history_probe._fit_logistic(
        x_train, y_train, x_test, y_test, epochs=200
    )
    scores = history_probe._fit_scores(
        x_train, y_train, x_test, epochs=200
    )
    assert scores is not None
    assert accuracy == float(((scores > 0).astype(np.float32) == y_test).mean())


def test_within_group_z_centres_each_group():
    scores = np.array([0.0, 2.0, 10.0, 14.0])
    keys = [b"a", b"a", b"b", b"b"]
    z = history_probe._within_group_z(scores, keys)
    np.testing.assert_allclose(z[:2], [-1.0, 1.0])
    np.testing.assert_allclose(z[2:], [-1.0, 1.0])


def test_within_group_z_constant_group_is_zero():
    scores = np.array([3.0, 3.0, 3.0])
    keys = [b"a", b"a", b"a"]
    z = history_probe._within_group_z(scores, keys)
    np.testing.assert_allclose(z, 0.0)


def test_card_ambiguous_keeps_states_without_card_evidence():
    # Group "a" has a flat card signal (z = -1 / +1); groups "b" and "c" are
    # constant, so their cards carry no evidence and stay in the subset.
    scores = np.array([0.0, 2.0, 5.0, 5.0, 5.0, 5.0])
    keys = [b"a", b"a", b"b", b"b", b"c", b"c"]
    mask = history_probe._card_ambiguous(scores, keys, 0.5)
    assert not mask[:2].any()
    assert mask[2:].all()


def test_card_ambiguous_threshold_is_inclusive():
    scores = np.array([0.0, 1.0])
    keys = [b"a", b"a"]
    assert history_probe._card_ambiguous(scores, keys, 1.0).all()
    assert not history_probe._card_ambiguous(scores, keys, 0.99).any()
