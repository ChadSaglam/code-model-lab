"""Tests for the pure statistics in scripts/analyze_results.py.

analyze_results.py turns experiment receipts into summary numbers and charts. The
charting is side-effect code, but the figures it draws rest on a handful of pure
functions that have carried no tests: the exact two-sided sign test that reports
whether greedy accuracy moved, the targeted family roll-ups, the paired per-task
differences fed to the bootstrap, and the bootstrap interval itself. A wrong number
here silently mis-states the headline result, so these pin the behaviour at its
boundaries (no observations, perfect symmetry, the clamp at p=1, empty input).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_results import (  # noqa: E402  (import after sys.path is set)
    bootstrap_interval,
    exact_two_sided_binomial,
    paired_task_differences,
    targeted_greedy_summary,
    targeted_sample_summary,
)


# ---------------------------------------------------------------- exact_two_sided_binomial


def test_binomial_no_observations_is_certain():
    # No task changed in either direction: there is nothing to reject, so p = 1.
    assert exact_two_sided_binomial(0, 0) == 1.0


def test_binomial_is_symmetric_in_gains_and_losses():
    # The test only knows the imbalance, not which side it favours.
    assert exact_two_sided_binomial(5, 0) == exact_two_sided_binomial(0, 5)
    assert exact_two_sided_binomial(7, 2) == exact_two_sided_binomial(2, 7)


def test_binomial_matches_the_closed_form_for_one_sided_splits():
    # With every observation on one side the two-sided p is 2 * (1/2)**n.
    assert exact_two_sided_binomial(5, 0) == 2 * 0.5**5  # 0.0625
    assert exact_two_sided_binomial(10, 0) == 2 * 0.5**10  # 0.001953125


def test_binomial_matches_the_closed_form_for_a_mixed_split():
    # n = 4, tail = 1: 2 * (C(4,0) + C(4,1)) / 2**4 = 2 * 5/16 = 0.625.
    assert exact_two_sided_binomial(3, 1) == 0.625


def test_binomial_clamps_at_one_when_the_split_is_near_even():
    # A balanced split would give 2 * p > 1; the result is capped at a probability.
    assert exact_two_sided_binomial(1, 1) == 1.0
    assert exact_two_sided_binomial(6, 5) == 1.0
    assert exact_two_sided_binomial(50, 50) == 1.0


def test_binomial_grows_less_significant_as_the_split_evens_out():
    # The more lopsided the split, the smaller (more significant) the p-value.
    assert exact_two_sided_binomial(10, 0) < exact_two_sided_binomial(8, 2)
    assert exact_two_sided_binomial(8, 2) < exact_two_sided_binomial(6, 4)
    assert exact_two_sided_binomial(6, 4) <= exact_two_sided_binomial(5, 5)


# ---------------------------------------------------------------- targeted_sample_summary


def _by_family(**families):
    return {"by_family": families}


def test_sample_summary_sums_only_the_target_families():
    payload = _by_family(
        increment={"exact_rollouts": 10, "rollouts": 20, "tasks_solved_at_k": 2, "tasks": 4},
        double={"exact_rollouts": 6, "rollouts": 20, "tasks_solved_at_k": 1, "tasks": 4},
        even={"exact_rollouts": 0, "rollouts": 20, "tasks_solved_at_k": 0, "tasks": 4},
        square={"exact_rollouts": 99, "rollouts": 99, "tasks_solved_at_k": 9, "tasks": 9},
    )
    summary = targeted_sample_summary(payload)
    assert summary["exact_rollouts"] == 16  # square is not a target and is excluded
    assert summary["rollouts"] == 60
    assert summary["exact_rollout_rate"] == 16 / 60
    assert summary["tasks_solved_at_8"] == 3
    assert summary["tasks"] == 12
    assert summary["pass_at_8"] == 3 / 12


# ---------------------------------------------------------------- targeted_greedy_summary


def _episode(task_id, family, passed, tests_passed, tests_total):
    return {
        "task_id": task_id,
        "family": family,
        "evaluation": {"passed": passed, "tests_passed": tests_passed, "tests_total": tests_total},
    }


def test_greedy_summary_counts_passes_and_tests_for_targets_only():
    payload = {
        "episodes": [
            _episode("a", "increment", True, 3, 3),
            _episode("b", "double", False, 1, 3),
            _episode("c", "even", True, 2, 3),
            _episode("d", "reverse", True, 3, 3),  # non-target, ignored
        ]
    }
    summary = targeted_greedy_summary(payload)
    assert summary["tasks"] == 3
    assert summary["tasks_solved"] == 2
    assert summary["pass_at_1"] == 2 / 3
    assert summary["tests_passed"] == 6  # 3 + 1 + 2
    assert summary["tests_total"] == 9


# ---------------------------------------------------------------- paired_task_differences


def _sampled_episode(task_id, family, passes):
    # `passes` is one bool per sample; the rest of the sample fields are unused here.
    return {
        "task_id": task_id,
        "family": family,
        "samples": [{"evaluation": {"passed": value}} for value in passes],
    }


def test_paired_differences_are_per_sample_and_target_only():
    before = {
        "episodes": [
            _sampled_episode("a", "increment", [True, False, False, False]),  # 1/4
            _sampled_episode("b", "double", [False, False, False, False]),  # 0/4
            _sampled_episode("z", "reverse", [True, True, True, True]),  # non-target
        ]
    }
    after = {
        "episodes": [
            _sampled_episode("a", "increment", [True, True, True, False]),  # 3/4
            _sampled_episode("b", "double", [False, False, False, False]),  # 0/4
            _sampled_episode("z", "reverse", [False, False, False, False]),  # non-target
        ]
    }
    diffs = paired_task_differences(before, after)
    # Driven by `after`'s target episodes, matched to `before` by task_id.
    assert diffs == [0.5, 0.0]  # a gained 2/4, b unchanged, reverse excluded


def test_paired_differences_can_be_negative_when_accuracy_regresses():
    before = {"episodes": [_sampled_episode("a", "even", [True, True, False, False])]}  # 2/4
    after = {"episodes": [_sampled_episode("a", "even", [False, False, False, False])]}  # 0/4
    assert paired_task_differences(before, after) == [-0.5]


# ---------------------------------------------------------------- bootstrap_interval


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    values = [0.1, -0.2, 0.3, 0.0, 0.5]
    first = bootstrap_interval(values, samples=2000, seed=123)
    second = bootstrap_interval(values, samples=2000, seed=123)
    assert first == second


def test_bootstrap_of_a_constant_sample_collapses_to_that_constant():
    low, high = bootstrap_interval([0.2, 0.2, 0.2], samples=1000)
    # Every resample draws only 0.2s, so both ends land on the constant (modulo float sum error).
    assert low == high == pytest.approx(0.2)


def test_bootstrap_interval_is_ordered_and_inside_the_value_range():
    values = [-0.4, -0.1, 0.0, 0.2, 0.6]
    low, high = bootstrap_interval(values, samples=5000, seed=7)
    assert low <= high
    assert min(values) <= low
    assert high <= max(values)
