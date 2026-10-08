"""Tests for lab/ladder.py: level parsing, custom-task loading and the family/level maps.

These are the pure-logic helpers the lab and the CLI runner lean on — `parse_levels` turns the
`--levels` flag into a validated list, `load_custom` reads level-5 tasks defensively, and
`family_levels`/`usable_levels` describe which families live at which level.
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


# ---------------------------------------------------------------- parse_levels

def test_parse_levels_sorts_and_deduplicates():
    assert ladder.parse_levels("3,1,2") == [1, 2, 3]
    assert ladder.parse_levels("1, 1 ,2") == [1, 2]  # whitespace tolerated, repeats collapsed


def test_parse_levels_single_level():
    assert ladder.parse_levels("4") == [4]


@pytest.mark.parametrize("text", ["", "   ", ",", " , "])
def test_parse_levels_rejects_empty_input(text):
    with pytest.raises(SystemExit, match="comma list of"):
        ladder.parse_levels(text)


def test_parse_levels_rejects_a_level_that_does_not_exist():
    with pytest.raises(SystemExit, match="comma list of"):
        ladder.parse_levels("9")


def test_parse_levels_rejects_level_five_until_custom_tasks_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "CUSTOM_FILE", tmp_path / "absent.json")
    assert 5 not in ladder.usable_levels()
    with pytest.raises(SystemExit, match="comma list of"):
        ladder.parse_levels("5")


def test_parse_levels_accepts_level_five_once_a_custom_task_is_added(monkeypatch, tmp_path):
    custom = tmp_path / "custom_tasks.json"
    custom.write_text(json.dumps([_custom_task()]))
    monkeypatch.setattr(ladder, "CUSTOM_FILE", custom)
    assert 5 in ladder.usable_levels()
    assert ladder.parse_levels("5") == [5]
    assert ladder.parse_levels("1,5") == [1, 5]


@pytest.mark.xfail(strict=True, reason="BUG: parse_levels leaks ValueError on a non-numeric token instead of the friendly SystemExit")
def test_parse_levels_rejects_non_numeric_token_with_a_friendly_error():
    with pytest.raises(SystemExit, match="comma list of"):
        ladder.parse_levels("abc")


# ---------------------------------------------------------------- load_custom

def _custom_task(name="triple"):
    return {
        "name": name,
        "arguments": "x",
        "descriptions": ["Return three times x.", "Triple x."],
        "body": "\n    return x * 3\n",
        "cases": [[[2], 6], [[0], 0]],
    }


def test_load_custom_builds_the_tuple_shape_the_course_expects(monkeypatch, tmp_path):
    custom = tmp_path / "custom_tasks.json"
    custom.write_text(json.dumps([_custom_task()]))
    monkeypatch.setattr(ladder, "CUSTOM_FILE", custom)
    rows = ladder.load_custom()
    assert len(rows) == 1
    name, arguments, descriptions, body, cases = rows[0]
    assert name == "triple"
    assert arguments == "x"
    assert descriptions == ("Return three times x.", "Triple x.")  # JSON lists become tuples
    assert body == "\n    return x * 3\n"
    assert cases == (((2,), 6), ((0,), 0))  # each case is (args-tuple, expected)


def test_load_custom_returns_empty_list_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "CUSTOM_FILE", tmp_path / "nope.json")
    assert ladder.load_custom() == []


def test_load_custom_returns_empty_list_on_malformed_json(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json")
    monkeypatch.setattr(ladder, "CUSTOM_FILE", bad)
    assert ladder.load_custom() == []


# ---------------------------------------------------------------- family/level maps

def test_family_levels_maps_each_family_to_its_level():
    mapping = ladder.family_levels()
    assert mapping["increment"] == 1
    assert mapping["add"] == 2
    assert mapping["sign"] == 3
    assert mapping["average"] == 4


def test_usable_levels_are_the_built_in_levels_without_custom_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(ladder, "CUSTOM_FILE", tmp_path / "absent.json")
    assert ladder.usable_levels() == [1, 2, 3, 4]


def test_family_rows_returns_rows_for_the_chosen_levels_in_level_order():
    rows = ladder.family_rows([2, 1])
    names = [row[0] for row in rows]
    # level 1 families come before level 2 families regardless of the argument order
    assert names[: len(ladder.LEVELS[1]["families"])] == [r[0] for r in ladder.LEVELS[1]["families"]]
    assert "add" in names and "increment" in names
