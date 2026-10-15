"""Tests for lab/ladder.py's stop rule: stops_at_function_end.

The course's stop rule (glm53_flash's completion_is_parseable) halts as soon as the source
parses, which would cut a multi-branch function off after its first branch. Task Lab needs the
stricter rule here: the completion must parse under the course verifier AND end on the final
4-space `return`. These tests pin that boundary behaviour — the parse gate, the trailing-newline
requirement and the exact indentation of the last line — on real tasks built from make_task.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lab" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ladder = load("ladder")

import glm53_flash.tasks as tasks  # noqa: E402

# Own Family objects so the global task list other tests mutate cannot leak in.
INCREMENT = tasks.Family(
    "increment", "x", ("Return x plus one.", "Increase x by one."),
    "\n    return x + 1\n", (((-3,), -2), ((0,), 1), ((7,), 8)))
ABSOLUTE = tasks.Family(
    "absolute", "x", ("Return the absolute value of x.", "Make a negative x positive."),
    "\n    return -x if x < 0 else x\n", (((-7,), 7), ((0,), 0), ((9,), 9)))


def task_for(family: tasks.Family) -> tasks.CodingTask:
    return tasks.make_task(family, split="confirm", index=0, seed=1)


# ---------------------------------------------------------------- accepts a finished function

def test_single_line_return_body_is_complete():
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    return x + 1\n") is True


def test_reference_completion_is_complete():
    task = task_for(ABSOLUTE)
    assert ladder.stops_at_function_end(task, task.reference_completion) is True


def test_multi_line_body_ending_on_the_final_return_is_complete():
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    y = x + 1\n    return y\n") is True


def test_trailing_blank_lines_after_the_return_are_ignored():
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    return x + 1\n\n") is True


# ---------------------------------------------------------------- not yet at the final return

def test_completion_stopped_inside_a_branch_is_not_complete():
    # parses fine, but the last line is a deeper-indented return: the function is not done.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    if x > 0:\n        return 1\n") is False


def test_last_line_that_is_not_a_return_is_not_complete():
    # a plain assignment parses as a (None-returning) function but is not the final return.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    y = x + 1\n") is False


def test_over_indented_final_return_is_not_complete():
    # the last line must sit at exactly four spaces; five spaces is not the function's own level.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n     return x + 1\n") is False


# ---------------------------------------------------------------- the trailing-newline requirement

def test_a_return_without_a_trailing_newline_is_not_complete():
    # identical body, only the closing newline missing: the stream has not finished the line.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    return x + 1") is False


# ---------------------------------------------------------------- the parse gate

def test_unparseable_completion_is_not_complete():
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    return (x +\n") is False


def test_empty_completion_is_not_complete():
    # prompt alone is a def header with no body: a SyntaxError, not a finished function.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "") is False


def test_well_formed_return_with_a_disallowed_call_is_not_complete():
    # the line looks final, but the course verifier rejects range(): the gate fails first.
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    return range(x)\n") is False


def test_disallowed_loop_is_not_complete():
    task = task_for(INCREMENT)
    assert ladder.stops_at_function_end(task, "\n    for i in x:\n        return i\n") is False
