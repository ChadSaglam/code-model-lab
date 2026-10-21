"""Tests for the pure statistics in scripts/analyze_rl_variants.py.

analyze_rl_variants.py screens several RL arms against one baseline on the same
fixed dev tasks. The report it writes rests on three pure functions that have
carried no tests: the exact two-sided McNemar sign test on discordant pairs, the
Holm step-down correction that keeps the reported p-values monotone, and the
seeded paired bootstrap interval. A wrong number here silently mis-ranks an arm
or over-states significance, so these pin the behaviour at its boundaries (no
discordant pairs, the clamp at p=1, Holm's running maximum, bootstrap of an
identical pair) and check the determinism the seed promises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_rl_variants import (  # noqa: E402  (import after sys.path is set)
    holm,
    mcnemar_exact,
    paired_bootstrap,
)


# ---------------------------------------------------------------- mcnemar_exact


def test_mcnemar_no_discordant_pairs_is_certain():
    # Every task agrees between arms: nothing moved, so there is nothing to reject.
    assert mcnemar_exact([True, False, True], [True, False, True]) == (0, 0, 1.0)


def test_mcnemar_counts_gains_and_losses_in_the_right_direction():
    # A gain is a task the left arm missed and the right arm solved; a loss is the
    # reverse. Here the left arm solves everything the right arm then loses.
    left = [True, True, True, True]
    right = [True, False, False, False]
    gains, losses, _ = mcnemar_exact(left, right)
    assert (gains, losses) == (0, 3)


def test_mcnemar_one_sided_streak_is_significant():
    # Five tasks all move the same way: p = 2 * C(5,0) / 2**5 = 2/32.
    assert mcnemar_exact([False] * 5, [True] * 5) == (5, 0, 0.0625)


def test_mcnemar_p_is_symmetric_in_gains_and_losses():
    # The test knows only the imbalance, not which arm it favours.
    _, _, p_right = mcnemar_exact([False] * 6, [True] * 6)
    _, _, p_left = mcnemar_exact([True] * 6, [False] * 6)
    assert p_right == p_left == 0.03125


def test_mcnemar_balanced_discordance_clamps_at_one():
    # One gain and one loss doubles past 1.0; the two-sided p is capped, not >1.
    assert mcnemar_exact([False, True], [True, False]) == (1, 1, 1.0)


def test_mcnemar_requires_equal_length_vectors():
    # A length mismatch means tasks are misaligned, not a silently truncated pair.
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


# ---------------------------------------------------------------- holm


def test_holm_single_comparison_is_unchanged():
    rows = [{"p_raw": 0.02}]
    holm(rows)
    assert rows[0]["p_holm"] == 0.02


def test_holm_scales_the_smallest_p_by_the_full_count():
    # Smallest raw p is multiplied by (count - rank) = 3 for three comparisons.
    rows = [{"p_raw": 0.01}, {"p_raw": 0.04}, {"p_raw": 0.03}]
    holm(rows)
    by_raw = {row["p_raw"]: row["p_holm"] for row in rows}
    assert by_raw[0.01] == pytest.approx(0.03)


def test_holm_enforces_a_monotone_running_maximum():
    # 0.04 alone would adjust to 0.04, but it may not drop below the 0.06 that the
    # earlier 0.03 reached, so the step-down holds it at 0.06.
    rows = [{"p_raw": 0.01}, {"p_raw": 0.04}, {"p_raw": 0.03}]
    holm(rows)
    by_raw = {row["p_raw"]: row["p_holm"] for row in rows}
    assert by_raw[0.03] == pytest.approx(0.06)
    assert by_raw[0.04] == pytest.approx(0.06)


def test_holm_clamps_each_adjusted_value_at_one():
    rows = [{"p_raw": 0.5}, {"p_raw": 0.6}]
    holm(rows)
    assert [row["p_holm"] for row in rows] == [1.0, 1.0]


def test_holm_adjusted_values_are_non_decreasing_in_raw_order():
    rows = [{"p_raw": p} for p in (0.001, 0.2, 0.01, 0.05)]
    holm(rows)
    ordered = sorted(rows, key=lambda row: row["p_raw"])
    adjusted = [row["p_holm"] for row in ordered]
    assert adjusted == sorted(adjusted)


# ---------------------------------------------------------------- paired_bootstrap


def test_bootstrap_of_an_identical_pair_has_zero_width():
    # Every per-task difference is zero, so every resample mean is zero too.
    vector = [True, False, True, False]
    assert paired_bootstrap(vector, vector) == (0.0, 0.0)


def test_bootstrap_interval_is_ordered_and_bounded():
    low, high = paired_bootstrap([False, False, True, True], [True, True, True, True])
    assert low <= high
    assert -1.0 <= low and high <= 1.0


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    left = [True, False, False, True, False]
    right = [True, True, False, True, True]
    assert paired_bootstrap(left, right) == paired_bootstrap(left, right)
