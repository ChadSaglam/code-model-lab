"""Behaviour tests for the fail-closed verifier in glm53_flash.evaluator.

validate_source is the sandbox that decides whether model- or user-written code is
allowed to run at all. The existing suite only checks that references pass and that an
empty body fails; these tests pin down the rejection rules and the type-strict scoring
that the whole Task Lab and Models leaderboard rely on.
"""
from __future__ import annotations

import ast

import pytest

from glm53_flash.evaluator import (
    completion_is_parseable,
    evaluate_source,
    validate_source,
)
from glm53_flash.tasks import CodingTask


def task(body: str, cases, *, entry: str = "f", arguments: str = "x") -> CodingTask:
    """A one-off task whose reference body and cases are supplied by the test."""
    prompt = f"def {entry}({arguments}):"
    return CodingTask(
        task_id="unit-fam-000",
        family="fam",
        entry_point=entry,
        prompt=prompt,
        reference_completion=body,
        cases=tuple(cases),
    )


# --- validate_source: what the sandbox accepts -----------------------------------

def test_valid_single_function_returns_module():
    tree = validate_source("def f(x):\n    return x + 1\n", "f")
    assert isinstance(tree, ast.Module)


def test_leading_comment_and_docstring_are_allowed():
    source = '# a comment\ndef f(x):\n    "doc"\n    return x\n'
    assert isinstance(validate_source(source, "f"), ast.Module)


def test_generator_expression_inside_sum_is_allowed():
    # The course rule bans for/while statements but explicitly permits sum(... for ...).
    source = "def f(values):\n    return sum(1 for v in values if v % 2 == 0)\n"
    assert isinstance(validate_source(source, "f"), ast.Module)


@pytest.mark.parametrize("name", ["sum", "len", "abs", "min", "max", "bool", "int", "str"])
def test_each_allowed_builtin_passes(name):
    assert isinstance(validate_source(f"def f(x):\n    return {name}(x)\n", "f"), ast.Module)


# --- validate_source: what the sandbox rejects -----------------------------------

@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n    for i in x:\n        return i\n",   # for loop
        "def f(x):\n    while x:\n        return x\n",      # while loop
        "def f(x):\n    import os\n    return x\n",         # import
        "def f(x):\n    from os import sep\n    return x\n",  # import-from
        "def f(x):\n    raise ValueError\n",               # raise
        "def f(x):\n    try:\n        return x\n    except Exception:\n        return 0\n",  # try
        "def f(x):\n    with x:\n        return x\n",       # with
        "def f(x):\n    global y\n    return x\n",          # global
        "def f(x):\n    del x\n    return 0\n",             # delete
        "def f(x):\n    g = lambda y: y\n    return g(x)\n",  # lambda
        "class C:\n    pass\ndef f(x):\n    return x\n",    # class at top level
    ],
)
def test_disallowed_syntax_is_rejected(source):
    with pytest.raises(ValueError):
        validate_source(source, "f")


def test_attribute_access_is_rejected():
    with pytest.raises(ValueError):
        validate_source("def f(text):\n    return text.upper()\n", "f")


def test_unknown_builtin_call_is_rejected():
    with pytest.raises(ValueError):
        validate_source("def f(x):\n    return print(x)\n", "f")


def test_dunder_name_is_rejected():
    with pytest.raises(ValueError):
        validate_source("def f(x):\n    return __import__\n", "f")


def test_recursion_is_rejected():
    with pytest.raises(ValueError):
        validate_source("def f(x):\n    return f(x)\n", "f")


def test_wrong_entry_point_name_is_rejected():
    with pytest.raises(ValueError):
        validate_source("def g(x):\n    return x\n", "f")


def test_extra_function_is_rejected():
    source = "def helper(x):\n    return x\ndef f(x):\n    return helper(x)\n"
    with pytest.raises(ValueError):
        validate_source(source, "f")


def test_extra_top_level_statement_is_rejected():
    source = "y = 1\ndef f(x):\n    return x + y\n"
    with pytest.raises(ValueError):
        validate_source(source, "f")


def test_oversized_source_is_rejected():
    # The 2048-byte cap bounds how much untrusted code is parsed and run.
    padding = "\n".join(f"    # {'z' * 40}" for _ in range(60))
    source = f"def f(x):\n{padding}\n    return x\n"
    assert len(source.encode("utf-8")) > 2048
    with pytest.raises(ValueError):
        validate_source(source, "f")


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        validate_source("def f(x):\n    return x +\n", "f")


# --- evaluate_source: scoring behaviour ------------------------------------------

def test_reference_passes_with_full_fraction():
    t = task("\n    return x + 1\n", [((0,), 1), ((7,), 8)])
    result = evaluate_source(t, "def f(x):\n    return x + 1\n")
    assert result.passed
    assert result.status == "passed"
    assert result.tests_passed == result.tests_total == 2
    assert result.pass_fraction == 1.0


def test_scoring_is_type_strict_bool_is_not_int():
    # bool(1) == 1 is True, but type(True) is not type(1): a wrong return type must fail.
    t = task("\n    return bool(x)\n", [((1,), 1)])
    result = evaluate_source(t, "def f(x):\n    return bool(x)\n")
    assert not result.passed
    assert result.tests_passed == 0


def test_partial_pass_reports_fraction():
    t = task("\n    return x\n", [((1,), 1), ((2,), 99)])
    result = evaluate_source(t, "def f(x):\n    return x\n")
    assert result.status == "failed"
    assert result.tests_passed == 1
    assert result.pass_fraction == 0.5


def test_runtime_error_in_candidate_counts_as_failure_not_crash():
    t = task("\n    return x\n", [((0,), 0)])
    result = evaluate_source(t, "def f(x):\n    return min(x)\n")  # min(int) raises TypeError at call
    assert not result.passed
    assert result.tests_passed == 0


def test_invalid_source_is_reported_not_raised():
    t = task("\n    return x\n", [((0,), 0)])
    result = evaluate_source(t, "def f(x):\n    import os\n    return x\n")
    assert result.status == "invalid"
    assert not result.passed
    assert result.pass_fraction == 0.0
    assert result.tests_total == 1  # the case count is still reported


# --- completion_is_parseable -----------------------------------------------------

def test_completion_parseable_requires_trailing_newline():
    t = task("\n    return x + 1\n", [((0,), 1)])
    assert completion_is_parseable(t, "\n    return x + 1\n")
    assert not completion_is_parseable(t, "\n    return x + 1")  # no trailing newline


def test_completion_not_parseable_when_invalid():
    t = task("\n    return x\n", [((0,), 0)])
    assert not completion_is_parseable(t, "\n    for i in x:\n        return i\n")
