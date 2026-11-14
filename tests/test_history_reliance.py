"""Tests for the street grouping in ``scripts/history_reliance.py``.

The rest of the script is a harness over the real environment; the part worth
pinning is how a mixed list of recorded decisions is split by street, because a
silent off-by-one there would move decisions between the reported rows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest

import history_reliance


def _row(street: int) -> tuple:
    """A recorded decision; only the street field matters to the grouping."""

    return (None, None, None, None, street, False)


def test_street_rows_group_counts_changes_and_shifts():
    recorded = [_row(0), _row(0), _row(1), _row(2), _row(3), _row(3)]
    changed = [True, False, True, False, False, True]
    shifts = [0.2, 0.4, 0.8, 0.1, 0.0, 0.6]

    rows = {name: (count, fraction, shift) for name, count, fraction, shift in
            history_reliance._street_rows(recorded, changed, shifts)}

    assert set(rows) == {"preflop", "flop", "turn", "river"}
    assert rows["preflop"][0] == 2
    assert rows["preflop"][1] == 0.5
    assert rows["preflop"][2] == pytest.approx(0.3)
    assert rows["flop"][:2] == (1, 1.0)
    assert rows["turn"][:2] == (1, 0.0)
    assert rows["river"][:2] == (2, 0.5)


def test_a_street_with_no_decisions_reports_nan():
    rows = history_reliance._street_rows([_row(0)], [True], [0.5])
    by_name = {name: (count, fraction, shift) for name, count, fraction, shift
               in rows}
    count, fraction, shift = by_name["river"]
    assert count == 0
    assert math.isnan(fraction) and math.isnan(shift)


def test_every_recorded_decision_lands_in_exactly_one_street():
    recorded = [_row(street) for street in (0, 0, 1, 2, 2, 3)]
    changed = [False] * len(recorded)
    shifts = [0.0] * len(recorded)
    rows = history_reliance._street_rows(recorded, changed, shifts)
    assert sum(count for _, count, _, _ in rows) == len(recorded)
