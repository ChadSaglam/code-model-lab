#!/usr/bin/env python3
"""Run a test set against a local Ollama model, with the same tasks and hidden tests as the small model.

    python lab/ollama_eval.py --model qwen3:8b --set course --output runs/local-models/<id>/result.json
    python lab/ollama_eval.py --model qwen3:8b --set levels --levels 1,2,3,4,5 --output ...

Prints one JSON line per task (the lab reads these for live progress) and writes a result file with
every answer, its test result and the tokens the model read and wrote.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder  # noqa: E402
import local_models  # noqa: E402

COURSE_FAMILIES = {"increment", "double", "even"}


def test_set(name: str, levels: list[int]):
    import glm53_flash.tasks as tasks
    if name == "course":  # exactly the 24 tasks of the Model Lab tests
        return [t for t in tasks.frozen_tasks("confirm", per_family=8) if t.family in COURSE_FAMILIES]
    ladder.activate(levels)  # exactly the tasks of a Task Lab test with these levels
    return tasks.frozen_tasks("confirm", per_family=4)


def summarize(episodes: list[dict], seconds: float) -> dict:
    written = sum(e["written"] for e in episodes)
    return {"tasks_solved": sum(e["passed"] for e in episodes), "tasks_total": len(episodes),
            "read": sum(e["read"] for e in episodes), "written": written, "seconds": round(seconds, 1),
            "tokens_per_second": round(written / max(1e-9, sum(e["seconds"] for e in episodes)), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--set", choices=("course", "levels"), required=True)
    parser.add_argument("--levels", default="1,2,3,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    levels = ladder.parse_levels(args.levels) if args.set == "levels" else [1]
    level_of = ladder.family_levels()
    tasks = test_set(args.set, levels)
    episodes, started = [], time.perf_counter()
    for task in tasks:
        reply = local_models.generate(args.model, local_models.build_prompt(task))
        code = local_models.extract_function(reply["text"], task.entry_point)
        result = local_models.check(task, code)
        episodes.append({"task": task.task_id, "family": task.family, "level": level_of.get(task.family, ladder.CUSTOM_LEVEL),
                         "prompt": task.prompt, "raw": reply["text"], "code": code, "passed": result["passed"],
                         "status": result["status"], "tests_passed": result["tests_passed"],
                         "tests_total": result["tests_total"], "read": reply["read"], "written": reply["written"],
                         "seconds": reply["seconds"]})
        print(json.dumps({"task": task.task_id, "passed": result["passed"], "read": reply["read"],
                          "written": reply["written"]}), flush=True)
    payload = {"schema_version": "1.0", "status": "complete", "model": args.model, "set": args.set,
               "levels": levels if args.set == "levels" else None, "summary": summarize(episodes, time.perf_counter() - started),
               "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except local_models.ModelError as error:
        print(f"Model error: {error}", flush=True)
        raise SystemExit(2)
