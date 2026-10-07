"""Non-code task kind: score a model's plain-text answer by rule, never in the sandbox.

Text tasks run no code. Where a coding task goes through the course verifier (validate_source in
glm53_flash/evaluator.py) and executes the model's function, a text task only compares the model's
plain-text answer against an expected value by one of three rules:

- exact    — the two strings match after whitespace is collapsed to single spaces;
- json     — the answer parses as a JSON object and the named fields equal the expected ones;
- numeric  — the answer parses as a number within a tolerance of the expected one.

A malformed answer (not JSON, not a number) scores wrong; it never raises.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

CHECKS = ("exact", "json", "numeric")


@dataclass(frozen=True)
class TextTask:
    task_id: str
    prompt: str
    check: str
    expected: Any
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.check not in CHECKS:
            raise ValueError(f"check must be one of {', '.join(CHECKS)}")


def _normalise(text: str) -> str:
    """Collapse every run of whitespace to a single space and trim the ends."""
    return " ".join(str(text).split())


def exact_match(expected: str, answer: str) -> bool:
    return _normalise(expected) == _normalise(answer)


def json_fields_match(expected: dict, answer: str) -> bool:
    try:
        parsed = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return False  # not JSON: scored wrong, not crashed
    if not isinstance(parsed, dict):
        return False
    return all(field in parsed and parsed[field] == value for field, value in expected.items())


def numeric_match(expected: float, answer: str, tolerance: float) -> bool:
    try:
        value = float(str(answer).strip())
    except (ValueError, TypeError):
        return False  # not a number: scored wrong, not crashed
    if not math.isfinite(value):
        return False
    return abs(value - expected) <= tolerance


def score(task: TextTask, answer: str) -> dict:
    """Score one plain-text answer against a text task. Mirrors local_models.check()'s result shape."""
    if task.check == "exact":
        ok = exact_match(task.expected, answer)
    elif task.check == "json":
        ok = json_fields_match(task.expected, answer)
    else:
        ok = numeric_match(task.expected, answer, task.tolerance)
    return {"status": "passed" if ok else "failed", "passed": ok, "message": ""}
