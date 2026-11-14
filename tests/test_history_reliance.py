"""Tests for the street grouping in ``scripts/history_reliance.py``.

The rest of the script is a harness over the real environment; the parts worth
pinning are how a mixed list of recorded decisions is split by street and how
the margin column is reduced, because a silent off-by-one there would move
decisions between the reported rows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import history_reliance


def _row(street: int) -> tuple:
    """A recorded decision; only the street field matters to the grouping."""

    return (None, None, None, None, street, False)


def test_street_rows_group_counts_changes_shifts_and_margins():
    recorded = [_row(0), _row(0), _row(1), _row(2), _row(3), _row(3)]
    changed = [True, False, True, False, False, True]
    shifts = [0.2, 0.4, 0.8, 0.1, 0.0, 0.6]
    margins = [0.1, 0.3, 1.5, 2.0, 0.5, 0.9]

    rows = {
        name: (count, fraction, shift, margin)
        for name, count, fraction, shift, margin in history_reliance._street_rows(
            recorded, changed, shifts, margins
        )
    }

    assert set(rows) == {"preflop", "flop", "turn", "river"}
    assert rows["preflop"][0] == 2
    assert rows["preflop"][1] == 0.5
    assert rows["preflop"][2] == pytest.approx(0.3)
    assert rows["preflop"][3] == pytest.approx(0.2)
    assert rows["flop"][:2] == (1, 1.0)
    assert rows["flop"][3] == pytest.approx(1.5)
    assert rows["turn"][:2] == (1, 0.0)
    assert rows["river"][:2] == (2, 0.5)
    assert rows["river"][3] == pytest.approx(0.7)


def test_a_street_with_no_decisions_reports_nan():
    rows = history_reliance._street_rows([_row(0)], [True], [0.5], [0.2])
    by_name = {
        name: (count, fraction, shift, margin)
        for name, count, fraction, shift, margin in rows
    }
    count, fraction, shift, margin = by_name["river"]
    assert count == 0
    assert math.isnan(fraction) and math.isnan(shift) and math.isnan(margin)


def test_every_recorded_decision_lands_in_exactly_one_street():
    recorded = [_row(street) for street in (0, 0, 1, 2, 2, 3)]
    changed = [False] * len(recorded)
    shifts = [0.0] * len(recorded)
    margins = [1.0] * len(recorded)
    rows = history_reliance._street_rows(recorded, changed, shifts, margins)
    assert sum(count for _, count, _, _, _ in rows) == len(recorded)
