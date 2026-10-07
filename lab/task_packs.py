"""Task-pack file format: a JSON bundle of text-answer tasks, validated on load.

A pack gathers many text tasks (the kinds from text_tasks.py) under one pack id and name, so a whole
benchmark — a set of bookkeeping or invoice tasks — loads from a single JSON file. The loader checks
the pack as it reads it, the way level-5 custom tasks are validated (lab/try_task.py validate,
lab/ladder.py load_custom): a malformed pack is rejected with a ValueError that names the offending
task id and field, never a bare stack trace from deep inside json. A valid pack becomes a TaskPack of
TextTask objects, ready to score with text_tasks.score.

The on-disk shape:

    {
      "id": "b1",
      "name": "Bookkeeping B1",
      "tasks": [
        {"id": "t1", "input": "...", "expected": "4400", "check": "exact"},
        {"id": "t2", "input": "...", "expected": {"vat": 7.7}, "check": "json"},
        {"id": "t3", "input": "...", "expected": 42.0, "check": "numeric", "tolerance": 0.5}
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from text_tasks import CHECKS, TextTask


@dataclass(frozen=True)
class TaskPack:
    pack_id: str
    name: str
    tasks: tuple[TextTask, ...]


def _require_str(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_task(raw, index: int, seen: set[str]) -> TextTask:
    where = f"task #{index + 1}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    for field in ("id", "input", "expected", "check"):
        if field not in raw:
            tid = raw.get("id") if isinstance(raw.get("id"), str) else where
            raise ValueError(f"task '{tid}' is missing required field '{field}'")
    task_id = _require_str(raw["id"], f"{where}: field 'id'")
    if task_id in seen:
        raise ValueError(f"task '{task_id}': duplicate task id")
    seen.add(task_id)
    prompt = _require_str(raw["input"], f"task '{task_id}': field 'input'")
    check = raw["check"]
    if check not in CHECKS:
        raise ValueError(f"task '{task_id}': field 'check' must be one of {', '.join(CHECKS)}")
    expected = raw["expected"]
    tolerance = raw.get("tolerance", 0.0)
    if check == "numeric":
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise ValueError(f"task '{task_id}': field 'expected' must be a number for a numeric check")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError(f"task '{task_id}': field 'tolerance' must be a number ≥ 0")
    else:
        if "tolerance" in raw:
            raise ValueError(f"task '{task_id}': field 'tolerance' applies only to a numeric check")
        if check == "json" and not isinstance(expected, dict):
            raise ValueError(f"task '{task_id}': field 'expected' must be an object for a json check")
        if check == "exact" and not isinstance(expected, str):
            raise ValueError(f"task '{task_id}': field 'expected' must be a string for an exact check")
    return TextTask(task_id, prompt, check, expected, float(tolerance))


def validate_pack(data) -> TaskPack:
    """Turn a parsed JSON pack into a TaskPack, raising ValueError on the first fault it finds."""
    if not isinstance(data, dict):
        raise ValueError("pack must be a JSON object")
    pack_id = _require_str(data.get("id"), "pack: field 'id'")
    name = _require_str(data.get("name"), "pack: field 'name'")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("pack: field 'tasks' must be a non-empty list")
    seen: set[str] = set()
    built = tuple(_validate_task(raw, index, seen) for index, raw in enumerate(tasks))
    return TaskPack(pack_id, name, built)


def load_pack(path: str | Path) -> TaskPack:
    """Read a pack file and validate it; raise ValueError with a clear reason if it cannot be used."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"pack file not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"pack file is not valid JSON: {error}") from None
    return validate_pack(data)


__all__ = ["TaskPack", "load_pack", "validate_pack"]
