"""Tests for ladder.lengths: the context and answer-token budgets a run is given.

lengths() turns the chosen task families into two budgets the training and generation
code depend on: ``sequence_length`` (how much context fits) and ``max_new_tokens`` (how
long an answer may be), plus the family count. Its contract is defensive — a floor on each
budget so even the tiniest task set stays runnable, a round-up of the context length to a
multiple of 32, and enough room that the longest prompt and its reference body always fit.
These boundaries are what matter: too small and a correct function is cut off mid-answer or
the prompt is truncated. The families are driven through the level-5 custom file so each
test controls the exact prompts and bodies lengths() measures.
"""
from __future__ import annotations

import importlib.util
import json
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


ladder = load("ladder")


def _task(name="f", arguments="x", descriptions=("Do a thing.", "Do it."),
          body="\n    return x\n", cases=(([1], 1),)):
    """One level-5 custom task in the on-disk JSON shape load_custom reads."""
    return {"name": name, "arguments": arguments, "descriptions": list(descriptions),
            "body": body, "cases": [list(case) for case in cases]}


@pytest.fixture()
def with_custom(monkeypatch, tmp_path):
    """Install a level-5 custom file and return a helper that sets its tasks, then lengths([5])."""
    custom = tmp_path / "custom_tasks.json"
    monkeypatch.setattr(ladder, "CUSTOM_FILE", custom)

    def budgets(tasks):
        custom.write_text(json.dumps(tasks))
        return ladder.lengths([ladder.CUSTOM_LEVEL])

    return budgets


def _longest_prompt_bytes(tasks):
    return max(len(ladder.PROMPT.format(description=d, name=t["name"], arguments=t["arguments"]).encode())
               for t in tasks for d in t["descriptions"])


def _longest_body_bytes(tasks):
    return max(len(t["body"].encode()) for t in tasks)


# ---------------------------------------------------------------- shape / families count

def test_lengths_reports_the_three_budget_keys(with_custom):
    size = with_custom([_task()])
    assert set(size) == {"sequence_length", "max_new_tokens", "families"}


def test_families_is_the_number_of_rows(with_custom):
    size = with_custom([_task(name="a"), _task(name="b"), _task(name="c")])
    assert size["families"] == 3


# ---------------------------------------------------------------- max_new_tokens (answer budget)

def test_max_new_tokens_floor_is_48_for_a_tiny_body(with_custom):
    # A one-line body is well under the floor, so the answer budget never drops below 48.
    size = with_custom([_task(body="\n    return x\n")])
    assert _longest_body_bytes([_task(body="\n    return x\n")]) + 24 < 48  # guard: the floor is what bites
    assert size["max_new_tokens"] == 48


def test_max_new_tokens_follows_a_large_body_above_the_floor(with_custom):
    big = "\n    total = x + x + x + x + x + x + x + x\n    return total + x + x\n"
    tasks = [_task(body=big)]
    size = with_custom(tasks)
    body_bytes = _longest_body_bytes(tasks)
    assert body_bytes + 24 > 48  # guard: this body clears the floor
    assert size["max_new_tokens"] == body_bytes + 24


def test_answer_budget_always_leaves_room_for_the_longest_body(with_custom):
    # Whatever the body, the answer budget exceeds it, so a complete function is never cut off.
    for body in ("\n    return x\n", "\n    return x + 1\n", "\n    return x * x * x * x * x\n"):
        tasks = [_task(body=body)]
        size = with_custom(tasks)
        assert size["max_new_tokens"] > _longest_body_bytes(tasks)


# ---------------------------------------------------------------- sequence_length (context budget)

def test_sequence_length_floor_is_128_for_a_small_task_set(with_custom):
    size = with_custom([_task()])
    assert size["sequence_length"] == 128


def test_sequence_length_is_always_a_multiple_of_32(with_custom):
    for description in ("short.", "a longer description " * 3, "x" * 200):
        size = with_custom([_task(descriptions=(description, description))])
        assert size["sequence_length"] % 32 == 0
        assert size["sequence_length"] >= 128


def test_sequence_length_grows_past_the_floor_for_a_long_prompt(with_custom):
    long_description = "explain the behaviour of this function in careful detail " * 4
    size = with_custom([_task(descriptions=(long_description, long_description))])
    assert size["sequence_length"] > 128


def test_sequence_length_holds_the_longest_prompt_body_and_bos(with_custom):
    # The window must fit the longest prompt plus a full body plus the one-byte start token.
    tasks = [_task(name="a", descriptions=("a tidy summary of the task", "do it"),
                   body="\n    return x\n"),
             _task(name="b", descriptions=("another much much longer prompt line " * 3,),
                   body="\n    step = x + x\n    return step + x + x + x\n")]
    size = with_custom(tasks)
    assert size["sequence_length"] >= _longest_prompt_bytes(tasks) + _longest_body_bytes(tasks) + 1


# ---------------------------------------------------------------- built-in levels

@pytest.mark.parametrize("levels", [[1], [4], [1, 2, 3, 4]])
def test_built_in_levels_satisfy_every_budget_invariant(monkeypatch, tmp_path, levels):
    # With no custom file only the built-in levels exist; each combination stays runnable.
    monkeypatch.setattr(ladder, "CUSTOM_FILE", tmp_path / "absent.json")
    size = ladder.lengths(levels)
    assert size["max_new_tokens"] >= 48
    assert size["sequence_length"] >= 128
    assert size["sequence_length"] % 32 == 0
    assert size["families"] == len(ladder.family_rows(levels))
