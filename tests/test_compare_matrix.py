"""Tests for the batch comparison driver in ``scripts/compare_matrix.py``.

The driver is mostly plumbing, so these tests pin the parts that would silently
produce a wrong table: which checkpoints each mode pairs, how seeds are
aggregated, and what happens when a run is missing.
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

import compare_matrix


def test_checkpoint_names_per_mode(tmp_path):
    history, control = compare_matrix._checkpoint_paths(
        "history-vs-control", 2, tmp_path
    )
    assert history.name == "agent_b_seed2_0050.pt"
    assert control.name == "agent_b_control_seed2_0050.pt"

    agent_a, agent_b = compare_matrix._checkpoint_paths("agent-a-vs-b", 1, tmp_path)
    assert agent_a.name == "agent_a_seed1_0050.pt"
    assert agent_b.name == "agent_b_seed1_0050.pt"

    on, off = compare_matrix._checkpoint_paths("ablation", 0, tmp_path)
    assert on == off


def test_checkpoint_names_follow_custom_prefixes(tmp_path):
    history, control = compare_matrix._checkpoint_paths(
        "history-vs-control", 2, tmp_path,
        history_prefix="agent_b_coin",
        control_prefix="agent_b_coin_control",
    )
    assert history.name == "agent_b_coin_seed2_0050.pt"
    assert control.name == "agent_b_coin_control_seed2_0050.pt"


def test_checkpoint_names_follow_the_requested_iteration(tmp_path):
    history, control = compare_matrix._checkpoint_paths(
        "history-vs-control", 1, tmp_path, iteration=100
    )
    assert history.name == "agent_b_seed1_0100.pt"
    assert control.name == "agent_b_control_seed1_0100.pt"

    on, off = compare_matrix._checkpoint_paths(
        "ablation", 0, tmp_path, iteration=30
    )
    assert on.name == "agent_b_seed0_0030.pt"
    assert on == off


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        compare_matrix._checkpoint_paths("nonsense", 0, tmp_path)


def test_aggregate_reports_mean_and_spread():
    mean, sd, half = compare_matrix._aggregate([0.1, 0.2, 0.3])
    assert mean == pytest.approx(0.2)
    assert sd == pytest.approx(0.1)
    assert half == pytest.approx(compare_matrix.T95_DF2 * 0.1 / np.sqrt(3))


def test_aggregate_with_one_seed_has_no_spread():
    mean, sd, half = compare_matrix._aggregate([0.25])
    assert mean == pytest.approx(0.25)
    assert np.isnan(sd) and np.isnan(half)


def test_missing_checkpoints_are_skipped_not_invented(tmp_path):
    missing = tmp_path / "agent_b_seed0_0050.pt"
    assert compare_matrix._policies(
        "history-vs-control", missing, missing, encoder=None
    ) is None


def test_main_writes_a_table_when_runs_are_missing(tmp_path, capsys):
    out = tmp_path / "table.md"
    code = compare_matrix.main(
        [
            "--hands", "2",
            "--seeds", "0",
            "--opponents", "passive",
            "--modes", "ablation",
            "--checkpoint-dir", str(tmp_path / "checkpoints"),
            "--out", str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "missing checkpoint" in printed
    assert out.read_text(encoding="utf-8").count("missing checkpoint") == 1
