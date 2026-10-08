#!/usr/bin/env python3
"""One task, one answer: run your own code, a local model or a saved snapshot against a task's tests.

The lab server starts this in a separate process (with a time limit) and sends one JSON request on stdin:
  {"mode": "check",    "family": "sign", "index": 0, "code": "def sign_...(x): ..."}
  {"mode": "ollama",   "family": "sign", "index": 0, "model": "qwen3:8b"}
  {"mode": "snapshot", "family": "sign", "index": 0, "checkpoint": "runs/.../checkpoint-0096"}
  {"mode": "validate", "description": "...", "code": "def name(x): ...", "tests": "3 -> 9\\n..."}
It prints one JSON answer on stdout.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder  # noqa: E402
import local_models  # noqa: E402

TASKS_PER_TYPE = 4


def find_task(family: str, index: int):
    import glm53_flash.tasks as tasks
    ladder.activate(ladder.usable_levels())
    matches = [t for t in tasks.frozen_tasks("confirm", per_family=TASKS_PER_TYPE) if t.family == family]
    if not matches or not 0 <= index < len(matches):
        raise ValueError("Unknown task.")
    return matches[index]


def answer(task, code: str, extra: dict) -> dict:
    return {"task": task.task_id, "prompt": task.prompt, "code": code, **local_models.check(task, code), **extra}


def ask_snapshot(task, checkpoint: str) -> dict:
    import torch
    from glm53_flash import ByteTokenizer
    from glm53_flash.runtime import generate_group, load_checkpoint
    path = (ROOT / checkpoint).resolve()
    if ROOT.resolve() not in path.parents or not (path / "model.pt").exists():
        raise ValueError("Unknown snapshot.")
    model = load_checkpoint(path, torch.device("cpu")).eval()
    tokenizer = ByteTokenizer()
    prompt_tokens = len(tokenizer.encode(task.prompt, bos=True))
    room = model.config.max_sequence_length - prompt_tokens
    level = ladder.family_levels().get(task.family, ladder.CUSTOM_LEVEL)
    budget = min(ladder.lengths([level])["max_new_tokens"], room)
    if budget < 8:
        raise ValueError("This snapshot's context is too short for this task. Pick a snapshot from a run that included this level.")
    generated = generate_group(model, tokenizer, task, group_size=1, max_new_tokens=budget,
                               temperature=1.0, sample=False, seed=2026)[0]
    return answer(task, task.prompt + generated["completion"],
                  {"raw": generated["completion"], "read": prompt_tokens, "written": generated["tokens"]})


def ask_ollama(task, model: str) -> dict:
    reply = local_models.generate(model, local_models.build_prompt(task))
    code = local_models.extract_function(reply["text"], task.entry_point)
    return answer(task, code, {"raw": reply["text"], "read": reply["read"], "written": reply["written"],
                               "seconds": reply["seconds"]})


def parse_tests(text: str) -> list[list]:
    """Lines like `3 -> 9`, `2, 5 -> 7` or `[1, 2] -> 3` (Python literals only)."""
    cases = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if "->" not in line:
            raise ValueError(f"Test line {number}: write it as  input -> expected  (for example  3 -> 9).")
        left, right = line.rsplit("->", 1)
        try:
            arguments = ast.literal_eval(f"({left.strip()},)")
            expected = ast.literal_eval(right.strip())
            json.dumps(expected)  # must survive being saved
        except (ValueError, SyntaxError, TypeError):
            raise ValueError(f"Test line {number}: use plain Python values such as 3, 'text', [1, 2] or True.") from None
        cases.append([list(arguments), expected])
    if len(cases) < 2:
        raise ValueError("Add at least two tests, one per line.")
    return cases


def validate(request: dict) -> dict:
    """Turn the page's form into a level-5 task, and prove the reference solution passes its own tests."""
    import glm53_flash.tasks as tasks
    description = " ".join(str(request.get("description", "")).split())
    if not 5 <= len(description) <= 90:
        raise ValueError("Describe the task in one sentence (5–90 characters).")
    code = str(request.get("code", "")).replace("\t", "    ").strip("\n") + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"The function has a syntax error on line {error.lineno}.") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Write exactly one function, starting with  def name(...):")
    function = tree.body[0]
    name = function.name
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name):
        raise ValueError("Use a short lowercase function name, like  total_price.")
    if name in ladder.family_levels() and name not in {row[0] for row in ladder.load_custom()}:
        raise ValueError(f"'{name}' is already a built-in task. Pick another name.")
    if function.args.defaults or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise ValueError("Use plain inputs only (no default values, *args or **kwargs).")
    arguments = ", ".join(argument.arg for argument in function.args.args)
    lines = code.split("\n")
    header_end = function.body[0].lineno - 1
    body = "\n" + "\n".join(lines[header_end:]).rstrip() + "\n"
    last = [line for line in body.split("\n") if line.strip()][-1]
    if not (last.startswith("    return") and not last.startswith("     ")):
        raise ValueError("The function's last line must be its final  return ...  (indented 4 spaces).")
    cases = parse_tests(str(request.get("tests", "")))
    family = tasks.Family(name, arguments, (description, description), body,
                          tuple((tuple(args), expected) for args, expected in cases))
    task = tasks.make_task(family, split="check", index=0, seed=0)
    result = local_models.check(task, task.reference_source)
    if not result["passed"]:
        detail = result["message"] or next((f"input {c['input']}: expected {c['expected']}, got {c['got']}"
                                            for c in result["cases"] if not c["ok"]), "")
        raise ValueError(f"Your function doesn't pass your own tests yet: {detail}")
    return {"task": {"name": name, "arguments": arguments, "descriptions": [description, description],
                     "body": body, "cases": cases}}


def main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    try:
        mode = request.get("mode")
        if mode == "validate":
            result = validate(request)
        else:
            task = find_task(str(request.get("family")), int(request.get("index", 0)))
            if mode == "show":
                result = {"task": task.task_id, "prompt": task.prompt, "entry_point": task.entry_point,
                          "reference": task.reference_source, "tests_total": len(task.cases)}
            elif mode == "check":
                result = answer(task, str(request.get("code", "")), {})
            elif mode == "ollama":
                result = ask_ollama(task, str(request.get("model", "")))
            elif mode == "snapshot":
                result = ask_snapshot(task, str(request.get("checkpoint", "")))
            else:
                raise ValueError("Unknown mode.")
        print(json.dumps({"ok": True, **result}))
    except (ValueError, local_models.ModelError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
