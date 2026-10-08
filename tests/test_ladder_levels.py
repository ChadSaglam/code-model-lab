"""Tests for ladder.py's level-selection logic: which families a level set contains and how.

ladder.py decides which task families a run sees. The runner's ``--levels`` string is parsed into a
sorted, de-duplicated list of in-range levels (parse_levels); the chosen levels are flattened into an
ordered family table (family_rows); every family is mapped back to its level (family_levels); and the
page's level cards come from describe(). Level 5 is read fresh from a JSON file (load_custom) that may
be missing or malformed. These are the pure pieces the server and runner both rely on, so their
boundaries — empty input, out-of-range levels, duplicates, an absent or broken custom file — matter.
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


ladder = load("ladder")


@pytest.fixture()
def no_custom(monkeypatch):
    """Point the level-5 file at a path that does not exist, so only the built-in levels are usable."""
    monkeypatch.setattr(ladder, "CUSTOM_FILE", ROOT / "lab" / "does-not-exist.json")
    return ladder


# ---------------------------------------------------------------- parse_levels: accepted

def test_parse_levels_keeps_a_sorted_in_range_list(no_custom):
    assert ladder.parse_levels("1,2,3") == [1, 2, 3]


def test_parse_levels_sorts_out_of_order_input(no_custom):
    assert ladder.parse_levels("3,1,2") == [1, 2, 3]


def test_parse_levels_drops_duplicates(no_custom):
    assert ladder.parse_levels("2,2,2") == [2]


def test_parse_levels_tolerates_spaces_and_empty_fields(no_custom):
    assert ladder.parse_levels("  1 , , 2 ") == [1, 2]


def test_parse_levels_accepts_a_single_level(no_custom):
    assert ladder.parse_levels("4") == [4]


# ---------------------------------------------------------------- parse_levels: rejected

def test_parse_levels_rejects_empty_text(no_custom):
    with pytest.raises(SystemExit, match=r"--levels"):
        ladder.parse_levels("")


def test_parse_levels_rejects_a_level_above_the_range(no_custom):
    with pytest.raises(SystemExit, match=r"--levels"):
        ladder.parse_levels("9")


def test_parse_levels_rejects_the_whole_list_if_one_level_is_out_of_range(no_custom):
    with pytest.raises(SystemExit):
        ladder.parse_levels("1,9")


def test_parse_levels_rejects_level_five_when_no_custom_tasks_exist(no_custom):
    # Level 5 only exists once a custom file provides families; without one it is not selectable.
    with pytest.raises(SystemExit):
        ladder.parse_levels("5")


# ---------------------------------------------------------------- family_rows / family_levels

def test_family_rows_for_one_level_is_that_levels_table(no_custom):
    assert ladder.family_rows([1]) == list(ladder.LEVELS[1]["families"])


def test_family_rows_concatenates_levels_in_sorted_order(no_custom):
    # Even passed high-to-low, the rows come out level 1 first, then level 3.
    rows = ladder.family_rows([3, 1])
    assert rows == list(ladder.LEVELS[1]["families"]) + list(ladder.LEVELS[3]["families"])


def test_family_levels_maps_each_family_to_its_level(no_custom):
    mapping = ladder.family_levels()
    assert mapping["increment"] == 1
    assert mapping["add"] == 2
    assert mapping["sign"] == 3
    assert mapping["average"] == 4


def test_family_levels_covers_every_built_in_family(no_custom):
    built_in = {row[0] for level in (1, 2, 3, 4) for row in ladder.LEVELS[level]["families"]}
    assert built_in.issubset(ladder.family_levels())


# ---------------------------------------------------------------- usable_levels

def test_usable_levels_are_the_built_in_four_without_custom_tasks(no_custom):
    assert ladder.usable_levels() == [1, 2, 3, 4]


def test_usable_levels_include_level_five_once_a_custom_task_is_present(monkeypatch, tmp_path):
    custom = tmp_path / "custom_tasks.json"
    custom.write_text('[{"name": "triple", "arguments": "x", "descriptions": ["Return 3x.", "Triple x."],'
                      ' "body": "\\n    return x * 3\\n", "cases": [[[2], 6], [[0], 0]]}]')
    monkeypatch.setattr(ladder, "CUSTOM_FILE", custom)
    assert ladder.usable_levels() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- load_custom

def test_load_custom_returns_empty_list_when_the_file_is_missing(no_custom):
    assert ladder.load_custom() == []


def test_load_custom_returns_empty_list_on_malformed_json(monkeypatch, tmp_path):
    broken = tmp_path / "custom_tasks.json"
    broken.write_text("{ not json")
    monkeypatch.setattr(ladder, "CUSTOM_FILE", broken)
    assert ladder.load_custom() == []


def test_load_custom_turns_json_rows_into_course_tuples(monkeypatch, tmp_path):
    custom = tmp_path / "custom_tasks.json"
    custom.write_text('[{"name": "triple", "arguments": "x", "descriptions": ["Return 3x.", "Triple x."],'
                      ' "body": "\\n    return x * 3\\n", "cases": [[[2], 6], [[0], 0]]}]')
    monkeypatch.setattr(ladder, "CUSTOM_FILE", custom)
    rows = ladder.load_custom()
    assert rows == [("triple", "x", ("Return 3x.", "Triple x."), "\n    return x * 3\n",
                     (((2,), 6), ((0,), 0)))]


# ---------------------------------------------------------------- describe

def test_describe_lists_every_level_with_its_family_names(no_custom):
    cards = {card["level"]: card for card in ladder.describe()}
    assert set(cards) == {1, 2, 3, 4, 5}
    assert cards[2]["families"] == [row[0] for row in ladder.LEVELS[2]["families"]]


def test_describe_takes_the_first_example_for_level_one_and_the_last_otherwise(no_custom):
    cards = {card["level"]: card for card in ladder.describe()}
    assert cards[1]["example"]["body"] == ladder.LEVELS[1]["families"][0][3]
    assert cards[3]["example"]["body"] == ladder.LEVELS[3]["families"][-1][3]


def test_describe_has_no_example_for_an_empty_level(no_custom):
    cards = {card["level"]: card for card in ladder.describe()}
    assert cards[5]["families"] == [] and cards[5]["example"] is None
