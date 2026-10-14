"""Behaviour tests for local_models.check() — the sandbox run and type-strict scoring.

check() is the one path that actually executes model- or user-written code: it validates the
source through the course verifier, runs it against the task's hidden tests, and reports every
case (input, expected, got, ok). The existing suite only confirms a reference passes and a loop
is rejected; these tests pin the result shape, the type-strict comparison (True is not 1, 2.0 is
not 2), runtime errors shown as a case result, and the invalid branch's counts and friendly text.
No model is contacted and nothing reaches Ollama.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from glm53_flash.tasks import CodingTask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lab" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


local_models = load("local_models")


def task(cases, *, entry: str = "f", arguments: str = "x") -> CodingTask:
    """A one-off coding task; check() only reads entry_point and cases."""
    prompt = f"def {entry}({arguments}):"
    return CodingTask(task_id="unit-fam-000", family="fam", entry_point=entry,
                      prompt=prompt, reference_completion="\n    return x\n", cases=tuple(cases))


# ---------------------------------------------------------------- the passing and failing shapes

def test_all_cases_pass_report_passed_with_no_message():
    result = local_models.check(task([((0,), 1), ((7,), 8)]), "def f(x):\n    return x + 1\n")
    assert result["status"] == "passed" and result["passed"] is True and result["message"] == ""
    assert result["tests_passed"] == 2 and result["tests_total"] == 2
    assert [case["ok"] for case in result["cases"]] == [True, True]


def test_partial_pass_counts_and_flags_each_case():
    # right for x == 0, wrong for x == 7 (returns 7, expected 8).
    result = local_models.check(task([((0,), 1), ((7,), 8)]), "def f(x):\n    return x + 1 if x == 0 else x\n")
    assert result["status"] == "failed" and result["passed"] is False
    assert result["tests_passed"] == 1 and result["tests_total"] == 2
    assert result["cases"][0]["ok"] is True
    assert result["cases"][1]["ok"] is False and result["cases"][1]["got"] == "7"


# ---------------------------------------------------------------- type-strict scoring

def test_a_bool_is_not_accepted_where_an_int_is_expected():
    # True == 1 in Python, but the grader compares type first: a bool must not score as the int 1.
    result = local_models.check(task([((5,), 1)]), "def f(x):\n    return x > 0\n")
    assert result["passed"] is False
    assert result["cases"][0]["ok"] is False and result["cases"][0]["got"] == "True"


def test_a_float_is_not_accepted_where_an_int_is_expected():
    # 2.0 == 2, yet an equal value of the wrong type is wrong.
    result = local_models.check(task([((2,), 2)]), "def f(x):\n    return x * 1.0\n")
    assert result["passed"] is False
    assert result["cases"][0]["ok"] is False and result["cases"][0]["got"] == "2.0"


def test_a_bool_result_is_accepted_where_a_bool_is_expected():
    # the strictness is symmetric: a genuine bool answer to a bool task still passes.
    result = local_models.check(task([((2,), True)]), "def f(x):\n    return x == 2\n")
    assert result["passed"] is True and result["cases"][0]["ok"] is True


# ---------------------------------------------------------------- errors inside the candidate

def test_a_runtime_error_becomes_that_cases_result_not_a_crash():
    result = local_models.check(task([((2,), 2)]), "def f(x):\n    return x + str(x)\n")
    assert result["status"] == "failed" and result["passed"] is False and result["tests_passed"] == 0
    assert result["cases"][0]["ok"] is False and result["cases"][0]["got"].startswith("TypeError")


# ---------------------------------------------------------------- the invalid branch

def test_disallowed_syntax_is_invalid_with_a_friendly_message_and_no_cases():
    result = local_models.check(task([((0,), 1), ((7,), 8)]),
                                "def f(x):\n    for v in [x]:\n        return v\n")
    assert result["status"] == "invalid" and result["passed"] is False
    assert result["tests_passed"] == 0 and result["tests_total"] == 2  # the task's own count, nothing ran
    assert result["cases"] == [] and "loop" in result["message"]


def test_a_missing_function_is_invalid_and_named_in_plain_words():
    result = local_models.check(task([((0,), 1)]), "def other(x):\n    return x\n")
    assert result["status"] == "invalid"
    assert "doesn't define the requested function" in result["message"]


# ---------------------------------------------------------------- how a case's input reads back

def test_case_input_reads_without_a_trailing_comma():
    single = local_models.check(task([((5,), 6)]), "def f(x):\n    return x + 1\n")
    assert single["cases"][0]["input"] == "5"
    pair = local_models.check(task([((1, 2), 3)], entry="g", arguments="a, b"),
                              "def g(a, b):\n    return a + b\n")
    assert pair["cases"][0]["input"] == "1, 2"
    text = local_models.check(task([(("ab",), "ab")], entry="h", arguments="s"),
                              "def h(s):\n    return s\n")
    assert text["cases"][0]["input"] == "'ab'"
