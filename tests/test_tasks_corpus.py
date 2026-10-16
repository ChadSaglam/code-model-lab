"""Behaviour tests for the synthetic coding corpus in glm53_flash/tasks.py.

Every evaluation in the lab is built on this generator: make_task turns a Family into a
CodingTask, frozen_tasks fans a split out into a deterministic set, and pretraining_text
emits reference sources for the pretraining stream. The existing suite only confirms that
the frozen references pass and that split identities are disjoint; it does not pin the
generator's own promises — determinism for a given (split, index, seed), the prompt and
task_id shape, the reference_source composition, split validation and counts, or that a
pretraining sample is valid Python. These tests do that. No model and no torch are touched.
"""
from __future__ import annotations

import ast

import pytest

from glm53_flash.tasks import (
    FAMILIES,
    FAMILY_BY_NAME,
    frozen_tasks,
    make_task,
    pretraining_text,
)

INCREMENT = FAMILY_BY_NAME["increment"]


# ---------------------------------------------------------------- make_task shape

def test_task_id_is_split_family_and_zero_padded_index():
    assert make_task(INCREMENT, split="dev", index=0, seed=1701).task_id == "dev-increment-000"
    assert make_task(INCREMENT, split="dev", index=12, seed=1701).task_id == "dev-increment-012"


def test_prompt_carries_the_header_a_family_description_and_the_signature():
    task = make_task(INCREMENT, split="dev", index=0, seed=1701)
    assert task.prompt.startswith("# Complete this Python function.\n# ")
    assert task.prompt.endswith(f"def {task.entry_point}({INCREMENT.arguments}):")
    assert any(description in task.prompt for description in INCREMENT.descriptions)


def test_entry_point_is_the_family_name_with_a_seven_letter_lowercase_suffix():
    entry = make_task(INCREMENT, split="dev", index=0, seed=1701).entry_point
    assert entry.startswith("increment_")
    suffix = entry.split("_", 1)[1]
    assert len(suffix) == 7 and suffix.isalpha() and suffix.islower()


def test_reference_source_is_the_prompt_followed_by_the_family_body():
    task = make_task(INCREMENT, split="dev", index=0, seed=1701)
    assert task.reference_completion == INCREMENT.body
    assert task.reference_source == task.prompt + INCREMENT.body


# ---------------------------------------------------------------- make_task determinism

def test_same_split_index_and_seed_reproduce_an_identical_task():
    first = make_task(INCREMENT, split="dev", index=0, seed=1701)
    second = make_task(INCREMENT, split="dev", index=0, seed=1701)
    assert first == second  # frozen dataclasses compare by value


def test_a_different_index_changes_the_generated_identity():
    first = make_task(INCREMENT, split="dev", index=0, seed=1701)
    second = make_task(INCREMENT, split="dev", index=1, seed=1701)
    assert first.entry_point != second.entry_point
    assert first.task_id != second.task_id


def test_a_different_seed_changes_the_generated_identity():
    first = make_task(INCREMENT, split="dev", index=0, seed=1701)
    second = make_task(INCREMENT, split="dev", index=0, seed=2909)
    assert first.entry_point != second.entry_point


# ---------------------------------------------------------------- frozen_tasks

def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split: nope"):
        frozen_tasks("nope")


@pytest.mark.parametrize("split", ["dev", "final", "rl", "confirm"])
def test_count_is_one_per_family_per_slot(split):
    assert len(frozen_tasks(split, per_family=1)) == len(FAMILIES)
    assert len(frozen_tasks(split, per_family=4)) == len(FAMILIES) * 4


def test_frozen_tasks_is_deterministic_across_calls():
    first = [task.task_id for task in frozen_tasks("dev")]
    second = [task.task_id for task in frozen_tasks("dev")]
    assert first == second


def test_task_ids_are_unique_within_a_split():
    ids = [task.task_id for task in frozen_tasks("final", per_family=4)]
    assert len(ids) == len(set(ids))


def test_make_task_reproduces_the_first_frozen_task_of_a_family():
    # frozen_tasks seeds the first family (increment) with the split's base seed; make_task
    # given the same (split, index, seed) must reproduce that task exactly.
    frozen_first = frozen_tasks("dev")[0]
    assert frozen_first.family == "increment" and frozen_first.task_id == "dev-increment-000"
    rebuilt = make_task(INCREMENT, split="dev", index=0, seed=1701)
    assert rebuilt.entry_point == frozen_first.entry_point
    assert rebuilt.prompt == frozen_first.prompt


# ---------------------------------------------------------------- pretraining_text

def test_pretraining_text_is_deterministic_for_a_given_index_and_seed():
    assert pretraining_text(0) == pretraining_text(0)
    assert pretraining_text(3, seed=42) == pretraining_text(3, seed=42)


def test_pretraining_text_varies_with_index_and_seed():
    assert pretraining_text(0) != pretraining_text(1)
    assert pretraining_text(0, seed=42) != pretraining_text(0, seed=7)


def test_pretraining_text_is_a_complete_parseable_reference_source():
    source = pretraining_text(5)
    assert source.startswith("# Complete this Python function.\n")
    ast.parse(source)  # a reference source is real Python, not a bare prompt


# ---------------------------------------------------------------- dataclass guarantees

def test_coding_task_is_frozen():
    task = make_task(INCREMENT, split="dev", index=0, seed=1701)
    with pytest.raises(Exception):  # FrozenInstanceError (a dataclasses subclass of it)
        task.task_id = "changed"  # type: ignore[misc]


def test_every_family_is_registered_by_name():
    assert set(FAMILY_BY_NAME) == {family.name for family in FAMILIES}
    assert all(FAMILY_BY_NAME[name].name == name for name in FAMILY_BY_NAME)
