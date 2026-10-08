"""Tests for try_task.parse_tests: the parser behind a level-5 task's example tests.

parse_tests reads the "input -> expected" lines a person types into the Try-a-task form and turns
them into saveable cases. It handles untrusted text, so its edge cases matter: it must accept plain
Python literals, skip blank lines, demand at least two tests, and reject anything that is not a
literal with a friendly, line-numbered message instead of a bare stack trace.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lab" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# try_task imports ladder and local_models by name, so load them first.
load("ladder")
load("local_models")
try_task = load("try_task")
parse_tests = try_task.parse_tests


# ---------------------------------------------------------------- accepted shapes


def test_single_argument_becomes_a_one_item_input_list():
    assert parse_tests("3 -> 9\n2 -> 4") == [[[3], 9], [[2], 4]]


def test_several_arguments_become_a_multi_item_input_list():
    assert parse_tests("2, 5 -> 7\n1, 1 -> 2") == [[[2, 5], 7], [[1, 1], 2]]


def test_list_string_bool_and_float_literals_are_kept():
    cases = parse_tests("[1, 2] -> 3\n'ab' -> 'AB'\nTrue -> False\n1.5 -> 3.0")
    assert cases == [[[[1, 2]], 3], [["ab"], "AB"], [[True], False], [[1.5], 3.0]]


def test_blank_and_whitespace_only_lines_are_skipped():
    assert parse_tests("3 -> 9\n\n2 -> 4\n   ") == [[[3], 9], [[2], 4]]


# ---------------------------------------------------------------- rejected input


def test_fewer_than_two_tests_is_rejected():
    with pytest.raises(ValueError, match="at least two tests"):
        parse_tests("3 -> 9")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="at least two tests"):
        parse_tests("")


def test_a_line_without_an_arrow_names_its_line_number():
    with pytest.raises(ValueError, match="Test line 1:"):
        parse_tests("3 9\n2 -> 4")


def test_the_reported_line_number_counts_the_offending_line():
    with pytest.raises(ValueError, match="Test line 2:"):
        parse_tests("3 -> 9\nx -> 4")


def test_a_bare_name_is_not_a_literal_and_is_rejected():
    with pytest.raises(ValueError, match="plain Python values"):
        parse_tests("x -> 9\n2 -> 4")


def test_a_function_call_is_not_a_literal_and_is_rejected():
    with pytest.raises(ValueError, match="plain Python values"):
        parse_tests("foo(1) -> 9\n2 -> 4")


def test_an_empty_left_side_is_rejected():
    with pytest.raises(ValueError, match="plain Python values"):
        parse_tests(" -> 9\n2 -> 4")


def test_only_the_last_arrow_splits_input_from_expected():
    # "1 -> 2 -> 3" leaves "1 -> 2" on the input side, which is not a literal.
    with pytest.raises(ValueError, match="plain Python values"):
        parse_tests("1 -> 2 -> 3\n4 -> 5")


# ---------------------------------------------------------------- known bug

def test_unsaveable_expected_value_is_rejected_with_a_friendly_error():
    # The expected value must survive json.dumps to be saved; a set literal parses but cannot be
    # serialised, so it should be rejected the same friendly way a non-literal is.
    with pytest.raises(ValueError):
        parse_tests("3 -> {1, 2}\n4 -> 5")
