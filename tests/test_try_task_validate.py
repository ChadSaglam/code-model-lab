"""Behaviour tests for try_task.validate: the gate that turns the Try-a-task form into a task.

validate is the boundary where a person's own function and tests become a level-5 task. Before any
code runs it enforces a chain of rules — the description length, exactly one function, a short
lowercase name that is not already built in, plain inputs only, a final return, at least two tests —
and only then proves the reference solution passes its own tests in the sandbox. parse_tests has its
own suite (test_try_task_parse.py); these tests pin validate's acceptance and every rejection path.
No model is contacted and nothing reaches Ollama.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lab" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# validate imports ladder and local_models by name, so load them first.
load("ladder")
load("local_models")
try_task = load("try_task")


def validate(*, description="triple a number", code="def triple(x):\n    return x * 3\n",
             tests="3 -> 9\n2 -> 6"):
    return try_task.validate({"mode": "validate", "description": description, "code": code, "tests": tests})


# ---------------------------------------------------------------- the accepted shape

def test_a_valid_task_round_trips_to_its_parts():
    result = validate()["task"]
    assert result["name"] == "triple"
    assert result["arguments"] == "x"
    assert result["descriptions"] == ["triple a number", "triple a number"]
    assert result["body"] == "\n    return x * 3\n"
    assert result["cases"] == [[[3], 9], [[2], 6]]


def test_a_two_argument_function_keeps_both_names_in_order():
    result = validate(description="add two numbers", code="def plus(a, b):\n    return a + b\n",
                      tests="2, 3 -> 5\n0, 0 -> 0")["task"]
    assert result["name"] == "plus"
    assert result["arguments"] == "a, b"


def test_the_description_whitespace_is_collapsed():
    result = validate(description="triple   the\tinput")["task"]
    assert result["descriptions"] == ["triple the input", "triple the input"]


def test_tabs_in_the_code_become_four_spaces_and_still_validate():
    result = validate(code="def triple(x):\n\treturn x * 3\n")["task"]
    assert result["body"] == "\n    return x * 3\n"


# ---------------------------------------------------------------- the description bounds

def test_a_description_under_five_characters_is_rejected():
    with pytest.raises(ValueError, match="one sentence"):
        validate(description="four")


def test_a_description_over_ninety_characters_is_rejected():
    with pytest.raises(ValueError, match="one sentence"):
        validate(description="d" * 91)


def test_a_description_that_is_only_whitespace_is_rejected():
    with pytest.raises(ValueError, match="one sentence"):
        validate(description="     ")


def test_a_description_of_exactly_five_characters_is_accepted():
    assert validate(description="three")["task"]["name"] == "triple"


# ---------------------------------------------------------------- one function, parseable

def test_a_syntax_error_names_its_line():
    with pytest.raises(ValueError, match="syntax error on line"):
        validate(code="def triple(x)\n    return x * 3\n")


def test_a_non_function_statement_is_rejected():
    with pytest.raises(ValueError, match="exactly one function"):
        validate(code="x = 1\n")


def test_two_functions_are_rejected():
    with pytest.raises(ValueError, match="exactly one function"):
        validate(code="def triple(x):\n    return x * 3\ndef extra(y):\n    return y\n")


# ---------------------------------------------------------------- the name rules

def test_an_uppercase_name_is_rejected():
    with pytest.raises(ValueError, match="lowercase"):
        validate(code="def Triple(x):\n    return x * 3\n")


def test_a_single_character_name_is_rejected():
    # the name pattern needs at least two characters.
    with pytest.raises(ValueError, match="lowercase"):
        validate(code="def t(x):\n    return x * 3\n")


def test_a_name_starting_with_an_underscore_is_rejected():
    # It parses fine, so it reaches the name rule, which demands a lowercase-letter first character.
    with pytest.raises(ValueError, match="lowercase"):
        validate(code="def _triple(x):\n    return x * 3\n")


def test_a_name_over_thirty_one_characters_is_rejected():
    name = "t" + "a" * 31  # 32 characters
    with pytest.raises(ValueError, match="lowercase"):
        validate(code=f"def {name}(x):\n    return x * 3\n")


def test_a_built_in_task_name_is_rejected():
    # 'double' is a level-1 family; a person cannot shadow it.
    with pytest.raises(ValueError, match="already a built-in task"):
        validate(code="def double(x):\n    return x * 3\n")


# ---------------------------------------------------------------- plain inputs only

def test_a_default_argument_is_rejected():
    with pytest.raises(ValueError, match="plain inputs only"):
        validate(code="def triple(x=1):\n    return x * 3\n")


def test_star_args_is_rejected():
    with pytest.raises(ValueError, match="plain inputs only"):
        validate(code="def triple(*xs):\n    return 0\n")


def test_a_keyword_only_argument_is_rejected():
    with pytest.raises(ValueError, match="plain inputs only"):
        validate(code="def triple(x, *, y):\n    return x\n")


# ---------------------------------------------------------------- the final return

def test_a_body_that_does_not_end_in_a_return_is_rejected():
    with pytest.raises(ValueError, match="final  return"):
        validate(code="def triple(x):\n    y = x * 3\n")


def test_a_last_line_return_nested_deeper_than_four_spaces_is_rejected():
    # The final return must sit at the function's own indent; a return only inside a block is refused.
    with pytest.raises(ValueError, match="final  return"):
        validate(code="def triple(x):\n    if x > 0:\n        return x * 3\n")


# ---------------------------------------------------------------- the tests and the self-check

def test_fewer_than_two_tests_is_surfaced_from_the_parser():
    with pytest.raises(ValueError, match="at least two tests"):
        validate(tests="3 -> 9")


def test_a_reference_that_fails_its_own_tests_is_rejected():
    # the function triples, but the tests claim 3 -> 10; the self-check must catch the mismatch.
    with pytest.raises(ValueError, match="doesn't pass your own tests"):
        validate(tests="3 -> 10\n2 -> 6")
