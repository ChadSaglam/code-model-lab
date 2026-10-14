"""Tests for text_tasks: the rule-based scorer for plain-text (non-code) answers (B-04).

A text task never runs code; it compares a model's plain-text answer against an expected
value by one of three rules — exact, json, numeric — and a malformed answer scores wrong
rather than raising. These tests pin each rule at its boundaries (whitespace, missing or
extra JSON fields, numeric ties, non-finite and non-parsable input) and the score() shape
that the rest of the lab reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))

from text_tasks import (  # noqa: E402  (import after sys.path is set)
    TextTask,
    exact_match,
    json_fields_match,
    numeric_match,
    score,
)


# ---------------------------------------------------------------- TextTask shape


def test_texttask_rejects_unknown_check():
    with pytest.raises(ValueError, match="check must be one of"):
        TextTask(task_id="t1", prompt="p", check="regex", expected="x")


@pytest.mark.parametrize("check", ["exact", "json", "numeric"])
def test_texttask_accepts_each_known_check(check):
    expected = {"a": 1} if check == "json" else ("1" if check == "exact" else 1.0)
    task = TextTask(task_id="t1", prompt="p", check=check, expected=expected)
    assert task.check == check
    assert task.tolerance == 0.0  # default when the caller gives none


# ---------------------------------------------------------------- exact_match


def test_exact_match_collapses_surrounding_and_inner_whitespace():
    assert exact_match("4400", "  4400 ") is True
    assert exact_match("hello world", "hello\tworld\n") is True
    assert exact_match("a b  c", "a   b c") is True


def test_exact_match_is_not_fuzzy():
    assert exact_match("4400", "4 400") is False  # a space inside digits is a different string
    assert exact_match("Konto", "konto") is False  # case matters
    assert exact_match("4400", "4400.0") is False


# ---------------------------------------------------------------- json_fields_match


def test_json_match_requires_every_expected_field_to_agree():
    assert json_fields_match({"vat": 7.7, "net": 100}, '{"vat": 7.7, "net": 100}') is True


def test_json_match_ignores_extra_fields_in_the_answer():
    assert json_fields_match({"vat": 7.7}, '{"vat": 7.7, "gross": 107.7}') is True


def test_json_match_fails_on_missing_or_wrong_field():
    assert json_fields_match({"vat": 7.7, "net": 100}, '{"vat": 7.7}') is False  # net missing
    assert json_fields_match({"vat": 7.7}, '{"vat": 8.1}') is False  # wrong value


def test_json_match_empty_expectation_accepts_any_object():
    assert json_fields_match({}, "{}") is True
    assert json_fields_match({}, '{"anything": 1}') is True


def test_json_match_rejects_non_object_and_malformed_json():
    assert json_fields_match({"a": 1}, "[1, 2, 3]") is False  # a JSON array is not an object
    assert json_fields_match({"a": 1}, "42") is False  # a bare number is not an object
    assert json_fields_match({"a": 1}, "not json at all") is False  # does not raise
    assert json_fields_match({"a": 1}, "") is False


# ---------------------------------------------------------------- numeric_match


def test_numeric_match_is_inclusive_at_the_tolerance_boundary():
    assert numeric_match(42.0, "42.5", tolerance=0.5) is True  # exactly on the edge counts
    assert numeric_match(42.0, "41.5", tolerance=0.5) is True
    assert numeric_match(42.0, "42.51", tolerance=0.5) is False  # just past the edge


def test_numeric_match_zero_tolerance_needs_an_exact_value():
    assert numeric_match(100.0, "100", tolerance=0.0) is True
    assert numeric_match(100.0, " 100.0 ", tolerance=0.0) is True  # whitespace is trimmed
    assert numeric_match(100.0, "100.01", tolerance=0.0) is False


def test_numeric_match_handles_zero_and_negative_values():
    assert numeric_match(0.0, "0", tolerance=0.0) is True
    assert numeric_match(-5.0, "-5", tolerance=0.0) is True
    assert numeric_match(-5.0, "-4.9", tolerance=0.2) is True


def test_numeric_match_rejects_non_finite_and_non_numeric_answers():
    assert numeric_match(1.0, "inf", tolerance=1e9) is False  # infinity never matches
    assert numeric_match(1.0, "nan", tolerance=1e9) is False  # nan never matches
    assert numeric_match(1.0, "abc", tolerance=1.0) is False  # does not raise
    assert numeric_match(1.0, "", tolerance=1.0) is False


# ---------------------------------------------------------------- score dispatch


def _task(check, expected, tolerance=0.0):
    return TextTask(task_id="t", prompt="p", check=check, expected=expected, tolerance=tolerance)


def test_score_passed_shape_mirrors_the_check_result():
    result = score(_task("exact", "4400"), "4400")
    assert result == {"status": "passed", "passed": True, "message": ""}


def test_score_failed_shape():
    result = score(_task("exact", "4400"), "9999")
    assert result == {"status": "failed", "passed": False, "message": ""}


def test_score_dispatches_to_the_json_rule():
    passed = score(_task("json", {"vat": 7.7}), '{"vat": 7.7}')
    failed = score(_task("json", {"vat": 7.7}), '{"vat": 2.5}')
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_score_dispatches_to_the_numeric_rule_with_tolerance():
    passed = score(_task("numeric", 42.0, tolerance=0.5), "42.4")
    failed = score(_task("numeric", 42.0, tolerance=0.5), "43.0")
    assert passed["status"] == "passed"
    assert failed["status"] == "failed"
