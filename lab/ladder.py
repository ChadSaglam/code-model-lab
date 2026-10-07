#!/usr/bin/env python3
"""Difficulty ladder for Task Lab: harder task families on top of the course's one-line tasks.

Level 1 is the course itself. Levels 2-4 add new families with hidden tests, all inside the
course verifier's safety rules (no loops, no imports, no attribute access).

Used two ways:
  * The lab server imports it for level information (no torch needed).
  * As a runner, it plugs the chosen levels into the unchanged course scripts:
      python lab/ladder.py --levels 1,2,3 -- scripts/train_pretrain.py --output ... (script args)
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_FILE = Path(__file__).resolve().parent / "custom_tasks.json"  # level 5: tasks you add on the page
CUSTOM_LEVEL = 5

# (name, arguments, (two task descriptions), reference body, ((args), expected) test cases)
LEVELS = {
    1: {"title": "One-liners", "about": "The course's tasks: one short line, one input.", "families": [
        ("increment", "x", ("Return x plus one.", "Increase x by one."), "\n    return x + 1\n", (((-3,), -2), ((0,), 1), ((7,), 8))),
        ("double", "x", ("Return two times x.", "Double the input."), "\n    return x * 2\n", (((-4,), -8), ((0,), 0), ((6,), 12))),
        ("square", "x", ("Return x squared.", "Multiply x by itself."), "\n    return x * x\n", (((-4,), 16), ((0,), 0), ((5,), 25))),
        ("absolute", "x", ("Return the absolute value of x.", "Make a negative x positive."), "\n    return -x if x < 0 else x\n", (((-7,), 7), ((0,), 0), ((9,), 9))),
        ("nonnegative", "x", ("Clamp x to at least zero.", "Return zero when x is negative."), "\n    return x if x > 0 else 0\n", (((-5,), 0), ((0,), 0), ((8,), 8))),
        ("even", "x", ("Return whether x is even.", "Check divisibility by two."), "\n    return x % 2 == 0\n", (((-3,), False), ((0,), True), ((8,), True))),
        ("reverse", "text", ("Return text in reverse order.", "Reverse the string."), "\n    return text[::-1]\n", ((("",), ""), (("abc",), "cba"), (("level",), "level"))),
        ("list_sum", "values", ("Return the sum of values.", "Add every number in the list."), "\n    return sum(values)\n", ((([],), 0), (([1, 2, 3],), 6), (([-2, 5],), 3))),
    ]},
    2: {"title": "Two steps", "about": "Two inputs or two operations in one line.", "families": [
        ("double_plus_one", "x", ("Return two times x plus one.", "Double x, then add one."), "\n    return x * 2 + 1\n", (((-2,), -3), ((0,), 1), ((5,), 11))),
        ("add", "a, b", ("Return the sum of a and b.", "Add a and b."), "\n    return a + b\n", (((2, 3), 5), ((-1, 1), 0), ((0, 0), 0))),
        ("larger", "a, b", ("Return the larger of a and b.", "Return whichever of a and b is bigger."), "\n    return max(a, b)\n", (((2, 7), 7), ((-3, -8), -3), ((4, 4), 4))),
        ("distance", "a, b", ("Return the distance between a and b.", "Return how far apart a and b are."), "\n    return abs(a - b)\n", (((2, 7), 5), ((7, 2), 5), ((-1, 1), 2))),
        ("last_item", "values", ("Return the last item of values.", "Return the final element of the list."), "\n    return values[-1]\n", ((([1, 2, 3],), 3), (([9],), 9), (([4, 0],), 0))),
        ("length", "text", ("Return the number of characters in text.", "Return the length of text."), "\n    return len(text)\n", ((("",), 0), (("abc",), 3), (("hi there",), 8))),
        ("max_of_three", "a, b, c", ("Return the largest of a, b and c.", "Return the biggest of three numbers."), "\n    return max(a, b, c)\n", (((1, 5, 3), 5), ((-1, -4, -2), -1), ((7, 7, 0), 7))),
    ]},
    3: {"title": "Conditions", "about": "if-statements across several lines, or a filter inside sum().", "families": [
        ("sign", "x", ("Return 1, -1 or 0 for the sign of x.", "Return the sign of x as 1, -1 or 0."),
         "\n    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0\n", (((5,), 1), ((-3,), -1), ((0,), 0))),
        ("clamp", "x", ("Limit x to the range 0 to 10.", "Keep x between 0 and 10."),
         "\n    if x < 0:\n        return 0\n    if x > 10:\n        return 10\n    return x\n", (((-4,), 0), ((15,), 10), ((7,), 7))),
        ("grade", "score", ("Return 'pass' if score is at least 50, else 'fail'.", "Grade score: 'pass' from 50, otherwise 'fail'."),
         "\n    if score >= 50:\n        return 'pass'\n    return 'fail'\n", (((70,), "pass"), ((50,), "pass"), ((20,), "fail"))),
        ("count_even", "values", ("Count the even numbers in values.", "Return how many values are even."),
         "\n    return sum(1 for v in values if v % 2 == 0)\n", ((([1, 2, 4],), 2), (([],), 0), (([3, 5],), 0))),
    ]},
    4: {"title": "Multi-step", "about": "Small functions of 3–7 lines with variables and several branches.", "families": [
        ("average", "values", ("Return the average of values, or 0 if empty.", "Return the mean of values; 0 for an empty list."),
         "\n    if len(values) == 0:\n        return 0\n    total = sum(values)\n    return total / len(values)\n", ((([],), 0), (([2, 4],), 3.0), (([1, 2],), 1.5))),
        ("fizzbuzz", "x", ("Return fizz, buzz or fizzbuzz for multiples of 3, 5 or both; else str(x).", "FizzBuzz: fizz for 3, buzz for 5, fizzbuzz for both, else str(x)."),
         "\n    if x % 15 == 0:\n        return 'fizzbuzz'\n    if x % 3 == 0:\n        return 'fizz'\n    if x % 5 == 0:\n        return 'buzz'\n    return str(x)\n",
         (((9,), "fizz"), ((10,), "buzz"), ((30,), "fizzbuzz"), ((7,), "7"))),
        ("spread", "values", ("Return the largest value minus the smallest value.", "Return the range of values."),
         "\n    low = min(values)\n    high = max(values)\n    return high - low\n", ((([3, 9, 1],), 8), (([5],), 0), (([-2, 4],), 6))),
        ("positive_sum", "values", ("Add only the positive numbers in values.", "Return the sum of the values above zero."),
         "\n    total = sum(v for v in values if v > 0)\n    return total\n", ((([1, -2, 3],), 4), (([],), 0), (([-1],), 0))),
    ]},
}

def load_custom() -> list[tuple]:
    """Level 5 rows from lab/custom_tasks.json (JSON lists become the tuples the course code expects)."""
    try:
        items = json.loads(CUSTOM_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [(t["name"], t["arguments"], tuple(t["descriptions"]), t["body"],
             tuple((tuple(args), expected) for args, expected in t["cases"])) for t in items]


def all_levels() -> dict[int, dict]:
    """The built-in levels plus level 5, read fresh so newly added tasks show up at once."""
    return {**LEVELS, CUSTOM_LEVEL: {"title": "Your tasks", "about": "Tasks you write yourself, checked by your own tests.",
                                     "families": load_custom()}}


def usable_levels() -> list[int]:
    return [level for level, info in all_levels().items() if info["families"]]


PROMPT = "# Complete this Python function.\n# {description}\ndef {name}_abcdefg({arguments}):"  # same shape as the course prompt


def family_rows(levels: list[int]) -> list[tuple]:
    table = all_levels()
    return [row for level in sorted(levels) for row in table[level]["families"]]


def family_levels() -> dict[str, int]:
    return {row[0]: level for level, info in all_levels().items() for row in info["families"]}


def lengths(levels: list[int]) -> dict[str, int]:
    """Context and answer budgets (in bytes = tokens) that fit every chosen family, with some slack."""
    rows = family_rows(levels)
    prompts = [len(PROMPT.format(description=d, name=r[0], arguments=r[1]).encode()) for r in rows for d in r[2]]
    bodies = [len(r[3].encode()) for r in rows]
    max_new_tokens = max(48, max(bodies) + 24)
    needed = max(max(prompts) + max_new_tokens, max(prompts) + max(bodies) + 2) + 1  # +BOS
    sequence_length = max(128, -(-needed // 32) * 32)
    return {"sequence_length": sequence_length, "max_new_tokens": max_new_tokens, "families": len(rows)}


def describe() -> list[dict]:
    """Level cards for the page: what each level contains, with one example task."""
    cards = []
    for level, info in all_levels().items():
        example = None
        if info["families"]:
            name, arguments, descriptions, body, _ = info["families"][0 if level == 1 else -1]
            example = {"prompt": PROMPT.format(description=descriptions[0], name=name, arguments=arguments), "body": body}
        cards.append({"level": level, "title": info["title"], "about": info["about"],
                      "families": [row[0] for row in info["families"]], "example": example})
    return cards


def stops_at_function_end(task, completion: str) -> bool:
    """Stop generating once the function is complete: it parses and its last line is the final 4-space return.

    The course rule stops as soon as the code parses, which would cut a multi-line function after its first
    branch. For one-line tasks both rules stop at the same place.
    """
    from glm53_flash.evaluator import validate_source
    try:
        validate_source(task.prompt + completion, task.entry_point)
    except (SyntaxError, ValueError):
        return False
    lines = [line for line in completion.split("\n") if line.strip()]
    return completion.endswith("\n") and bool(lines) and lines[-1].startswith("    return") and not lines[-1].startswith("     ")


def activate(levels: list[int]) -> None:
    """Swap the course's task list for the chosen levels, in the already-imported course modules."""
    import glm53_flash.evaluator as evaluator
    import glm53_flash.runtime as runtime
    import glm53_flash.tasks as tasks
    families = tuple(tasks.Family(*row) for row in family_rows(levels))
    tasks.FAMILIES = families
    tasks.FAMILY_BY_NAME = {family.name: family for family in families}
    evaluator.completion_is_parseable = stops_at_function_end
    runtime.completion_is_parseable = stops_at_function_end


def parse_levels(text: str) -> list[int]:
    levels = sorted({int(value) for value in text.split(",") if value.strip()})
    if not levels or any(level not in usable_levels() for level in levels):
        raise SystemExit(f"--levels must be a comma list of {usable_levels()}")
    return levels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a course script with Task Lab difficulty levels.")
    parser.add_argument("--levels", required=True)
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    activate(parse_levels(options.levels))
    script = str((ROOT / options.script).resolve())
    sys.argv = [script, *[arg for arg in options.args if arg != "--"]]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
