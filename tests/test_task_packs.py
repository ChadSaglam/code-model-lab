"""Tests for task_packs.validate_pack / load_pack: the task-pack loader (B-04).

A pack is an untrusted JSON bundle of text tasks. The loader must turn a valid pack into a
TaskPack of TextTask objects and reject a malformed one with a ValueError that names the
offending field — never a bare stack trace from deep inside json. These tests pin that
behaviour: the accepted shape, and every rejection branch (missing fields, duplicate ids,
per-check type rules, tolerance rules, empty input) plus the two file-level faults in load_pack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))

import task_packs  # noqa: E402  (import after sys.path is set)
from text_tasks import TextTask  # noqa: E402

validate_pack = task_packs.validate_pack
load_pack = task_packs.load_pack
TaskPack = task_packs.TaskPack


def pack(tasks, pack_id="b1", name="Bookkeeping B1"):
    return {"id": pack_id, "name": name, "tasks": tasks}


# ---------------------------------------------------------------- accepted shape


def test_valid_pack_becomes_tasks_of_each_kind():
    result = validate_pack(
        pack(
            [
                {"id": "t1", "input": "add it", "expected": "4400", "check": "exact"},
                {"id": "t2", "input": "vat?", "expected": {"vat": 7.7}, "check": "json"},
                {"id": "t3", "input": "sum?", "expected": 42.0, "check": "numeric", "tolerance": 0.5},
            ]
        )
    )
    assert isinstance(result, TaskPack)
    assert result.pack_id == "b1"
    assert result.name == "Bookkeeping B1"
    assert len(result.tasks) == 3
    assert all(isinstance(t, TextTask) for t in result.tasks)
    assert [t.task_id for t in result.tasks] == ["t1", "t2", "t3"]
    assert result.tasks[0].check == "exact"
    assert result.tasks[2].tolerance == 0.5


def test_numeric_tolerance_defaults_to_zero_as_float():
    result = validate_pack(pack([{"id": "t1", "input": "n", "expected": 5, "check": "numeric"}]))
    tol = result.tasks[0].tolerance
    assert tol == 0.0
    assert isinstance(tol, float)


def test_integer_expected_is_accepted_for_numeric():
    result = validate_pack(pack([{"id": "t1", "input": "n", "expected": 10, "check": "numeric"}]))
    assert result.tasks[0].expected == 10


# ---------------------------------------------------------------- pack-level faults


def test_pack_must_be_an_object():
    with pytest.raises(ValueError, match="pack must be a JSON object"):
        validate_pack([])


@pytest.mark.parametrize("missing", ["id", "name"])
def test_pack_id_and_name_must_be_non_empty_strings(missing):
    data = pack([{"id": "t1", "input": "n", "expected": "a", "check": "exact"}])
    data[missing] = "   "
    with pytest.raises(ValueError, match=f"pack: field '{missing}' must be a non-empty string"):
        validate_pack(data)


def test_tasks_must_be_a_non_empty_list():
    with pytest.raises(ValueError, match="field 'tasks' must be a non-empty list"):
        validate_pack(pack([]))


# ---------------------------------------------------------------- task-level faults


def test_task_must_be_an_object():
    with pytest.raises(ValueError, match="task #1 must be an object"):
        validate_pack(pack(["not-a-dict"]))


def test_missing_required_field_names_the_task_and_field():
    with pytest.raises(ValueError, match="task 't1' is missing required field 'expected'"):
        validate_pack(pack([{"id": "t1", "input": "n", "check": "exact"}]))


def test_duplicate_task_id_is_rejected():
    tasks = [
        {"id": "dup", "input": "a", "expected": "x", "check": "exact"},
        {"id": "dup", "input": "b", "expected": "y", "check": "exact"},
    ]
    with pytest.raises(ValueError, match="task 'dup': duplicate task id"):
        validate_pack(pack(tasks))


def test_empty_task_id_is_rejected():
    with pytest.raises(ValueError, match="field 'id' must be a non-empty string"):
        validate_pack(pack([{"id": "  ", "input": "n", "expected": "a", "check": "exact"}]))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="task 't1': field 'input' must be a non-empty string"):
        validate_pack(pack([{"id": "t1", "input": "", "expected": "a", "check": "exact"}]))


def test_unknown_check_is_rejected():
    with pytest.raises(ValueError, match="field 'check' must be one of"):
        validate_pack(pack([{"id": "t1", "input": "n", "expected": "a", "check": "regex"}]))


# ---------------------------------------------------------------- per-check type rules


def test_numeric_expected_must_be_a_number():
    with pytest.raises(ValueError, match="field 'expected' must be a number for a numeric check"):
        validate_pack(pack([{"id": "t1", "input": "n", "expected": "nan", "check": "numeric"}]))


def test_numeric_expected_rejects_bool():
    with pytest.raises(ValueError, match="field 'expected' must be a number for a numeric check"):
        validate_pack(pack([{"id": "t1", "input": "n", "expected": True, "check": "numeric"}]))


def test_negative_tolerance_is_rejected():
    tasks = [{"id": "t1", "input": "n", "expected": 1.0, "check": "numeric", "tolerance": -0.5}]
    with pytest.raises(ValueError, match="field 'tolerance' must be a number"):
        validate_pack(pack(tasks))


def test_bool_tolerance_is_rejected():
    tasks = [{"id": "t1", "input": "n", "expected": 1.0, "check": "numeric", "tolerance": True}]
    with pytest.raises(ValueError, match="field 'tolerance' must be a number"):
        validate_pack(pack(tasks))


def test_tolerance_only_applies_to_numeric_checks():
    tasks = [{"id": "t1", "input": "n", "expected": "a", "check": "exact", "tolerance": 1.0}]
    with pytest.raises(ValueError, match="field 'tolerance' applies only to a numeric check"):
        validate_pack(pack(tasks))


def test_json_expected_must_be_an_object():
    with pytest.raises(ValueError, match="field 'expected' must be an object for a json check"):
        validate_pack(pack([{"id": "t1", "input": "n", "expected": "not-json", "check": "json"}]))


def test_exact_expected_must_be_a_string():
    with pytest.raises(ValueError, match="field 'expected' must be a string for an exact check"):
        validate_pack(pack([{"id": "t1", "input": "n", "expected": 5, "check": "exact"}]))


# ---------------------------------------------------------------- load_pack (file level)


def test_load_pack_reads_and_validates_a_file(tmp_path):
    path = tmp_path / "pack.json"
    path.write_text('{"id":"b1","name":"N","tasks":[{"id":"t1","input":"n","expected":"a","check":"exact"}]}')
    result = load_pack(path)
    assert isinstance(result, TaskPack)
    assert result.tasks[0].task_id == "t1"


def test_load_pack_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="pack file not found"):
        load_pack(tmp_path / "nope.json")


def test_load_pack_invalid_json_gives_a_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="pack file is not valid JSON"):
        load_pack(path)


# ---------------------------------------------------------------- B1 bookkeeping pack (B-05)

B1_PACK_PATH = ROOT / "packs" / "b1_bookkeeping.json"


@pytest.fixture(scope="module")
def b1_pack():
    return load_pack(B1_PACK_PATH)


def test_b1_pack_loads_and_validates(b1_pack):
    assert isinstance(b1_pack, TaskPack)
    assert b1_pack.pack_id == "b1"


def test_b1_pack_task_count_is_within_range(b1_pack):
    assert 60 <= len(b1_pack.tasks) <= 100


def test_b1_pack_all_checks_are_exact(b1_pack):
    assert all(t.check == "exact" for t in b1_pack.tasks)


def test_b1_pack_expected_values_are_plausible_account_numbers(b1_pack):
    for t in b1_pack.tasks:
        assert isinstance(t.expected, str), f"{t.task_id}: expected must be a string"
        # KMU-Kontenrahmen accounts are four-digit numbers like 6500, 6200, 4400.
        assert t.expected.isdigit() and len(t.expected) == 4, f"{t.task_id}: {t.expected!r}"


def test_b1_pack_task_ids_are_unique(b1_pack):
    ids = [t.task_id for t in b1_pack.tasks]
    assert len(ids) == len(set(ids))
