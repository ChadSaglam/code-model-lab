"""Shared helpers for asking models and checking code: Ollama client, answer extraction, safe test runs.

Everything that runs model-written or user-written code goes through check(): the course verifier's
rules (no imports, loops, attribute access or unknown calls) plus a time limit per task.
"""
from __future__ import annotations

import json
import os
import re
import signal
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = f"http://{OLLAMA}"
TIME_LIMIT_SECONDS = 5

# The test rules go into the prompt, not the system message, so a model's own system message
# (for example a variant made on the Models page) still applies.
RULES = (
    "Complete small Python functions for an automated test. Reply with ONLY the complete function "
    "definition as plain Python code: no explanation and no markdown. The test rejects code that uses "
    "loops (for/while), imports, method calls or attribute access (like .split()), or extra functions. "
    "Only these built-ins are allowed: sum, len, abs, min, max, bool, int, str. "
    "Generator expressions inside sum() are allowed."
)


class ModelError(Exception):
    """Ollama is unreachable or answered with an error."""


def _post(path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(f"{OLLAMA}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise ModelError(error.read().decode(errors="replace")[:300] or str(error)) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise ModelError(f"Ollama is not reachable at {OLLAMA} ({error}). Is the Ollama app running?") from None


def model_details(timeout: float = 3) -> list[dict]:
    """Installed models with size, family, parameter count and quantization, as Ollama reports them."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=timeout) as response:
            models = json.loads(response.read()).get("models", [])
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
        raise ModelError(f"Ollama is not reachable at {OLLAMA}. Start the Ollama app to test local models.") from error
    rows = []
    for model in models:
        details = model.get("details") or {}
        rows.append({"name": model["name"], "size": model.get("size"), "family": details.get("family"),
                     "parameters": details.get("parameter_size"), "quantization": details.get("quantization_level"),
                     "modified": model.get("modified_at")})
    return sorted(rows, key=lambda row: row["name"])


def list_models(timeout: float = 3) -> list[str]:
    return [row["name"] for row in model_details(timeout)]


def create_variant(name: str, base: str, system: str, timeout: float = 600) -> None:
    """A new Ollama model that reuses the base model's weights with its own instructions (no training)."""
    try:
        _post("/api/create", {"model": name, "from": base, "system": system, "stream": False}, timeout)
    except ModelError as error:
        if "modelfile" not in str(error).lower() and "from" not in str(error).lower():
            raise
        quoted = system.replace('"' * 3, "'" * 3)  # older Ollama versions only accept a Modelfile
        modelfile = f'FROM {base}\nSYSTEM {chr(34) * 3}{quoted}{chr(34) * 3}\n'
        _post("/api/create", {"model": name, "modelfile": modelfile, "stream": False}, timeout)


def delete_model(name: str, timeout: float = 60) -> None:
    request = urllib.request.Request(f"{OLLAMA}/api/delete", data=json.dumps({"model": name}).encode(),
                                     headers={"Content-Type": "application/json"}, method="DELETE")
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
    except urllib.error.HTTPError as error:
        raise ModelError(error.read().decode(errors="replace")[:300] or str(error)) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise ModelError(f"Ollama is not reachable at {OLLAMA}.") from error


def generate(model: str, prompt: str, *, max_tokens: int = 512, timeout: float = 300) -> dict:
    """One deterministic answer. Thinking is switched off where the model supports it."""
    payload = {"model": model, "prompt": prompt, "stream": False, "think": False,
               "options": {"temperature": 0, "num_predict": max_tokens}}
    started = time.perf_counter()
    try:
        data = _post("/api/generate", payload, timeout)
    except ModelError as error:
        if "think" not in str(error).lower():
            raise
        payload.pop("think")  # models without a thinking mode reject the option
        data = _post("/api/generate", payload, timeout)
    return {"text": data.get("response", ""), "read": data.get("prompt_eval_count", 0),
            "written": data.get("eval_count", 0), "seconds": round(time.perf_counter() - started, 2)}


def build_prompt(task) -> str:
    return f"{RULES}\n\nComplete this Python function and reply with the whole function:\n\n{task.prompt}\n"


def extract_function(text: str, entry_point: str) -> str:
    """Pull the requested function out of a chat-style answer (thinking, markdown fences, explanations)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if fenced:
        text = next((block for block in fenced if f"def {entry_point}" in block), fenced[0])
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if line.lstrip().startswith(f"def {entry_point}")), None)
    if start is None:
        return text.strip() + "\n"
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start][indent:]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break  # back at the def's own level: the function has ended
        body.append(line[indent:] if line.strip() else "")
    return "\n".join(body).rstrip() + "\n"


FRIENDLY = [
    ("unsafe call", "calls a function that isn't allowed (only sum, len, abs, min, max, bool, int, str)"),
    ("attribute access", "uses .something (method calls and attributes aren't allowed)"),
    ("disallowed syntax: For", "uses a for loop (loops aren't allowed; try sum(... for ...))"),
    ("disallowed syntax: While", "uses a while loop (loops aren't allowed)"),
    ("disallowed syntax: Import", "imports a module (imports aren't allowed)"),
    ("disallowed syntax", "uses Python syntax the tests don't allow"),
    ("required function missing", "doesn't define the requested function (keep the name from the task)"),
    ("recursion", "calls itself (recursion isn't allowed)"),
]


def friendly(message: str) -> str:
    return next((text for key, text in FRIENDLY if key in message), message)


class _TimeLimit:
    def __enter__(self):
        if hasattr(signal, "SIGALRM"):  # macOS and Linux; Windows runs without the limit
            signal.signal(signal.SIGALRM, self._expired)
            signal.alarm(TIME_LIMIT_SECONDS)
        return self

    def __exit__(self, *exc):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return False

    @staticmethod
    def _expired(*_):
        raise TimeoutError(f"took longer than {TIME_LIMIT_SECONDS} s")


def check(task, source: str) -> dict:
    """Run one source against a task's tests and report every case (input, expected, got)."""
    from glm53_flash.evaluator import SAFE_BUILTINS, validate_source
    cases = []
    try:
        with _TimeLimit():
            tree = validate_source(source, task.entry_point)
            namespace = {"__builtins__": SAFE_BUILTINS}
            exec(compile(tree, "candidate.py", "exec"), namespace)
            function = namespace[task.entry_point]
    except (SyntaxError, ValueError, TypeError, KeyError, TimeoutError) as error:
        return {"status": "invalid", "passed": False, "message": friendly(str(error)), "tests_passed": 0,
                "tests_total": len(task.cases), "cases": []}
    for arguments, expected in task.cases:
        try:
            with _TimeLimit():
                actual = function(*arguments)
            ok = type(actual) is type(expected) and actual == expected
            cases.append({"input": repr(arguments)[1:-1].rstrip(","), "expected": repr(expected), "got": repr(actual), "ok": ok})
        except Exception as error:  # the candidate's own error, shown as its result
            cases.append({"input": repr(arguments)[1:-1].rstrip(","), "expected": repr(expected),
                          "got": f"{type(error).__name__}: {error}", "ok": False})
    passed = sum(case["ok"] for case in cases)
    return {"status": "passed" if passed == len(cases) else "failed", "passed": passed == len(cases), "message": "",
            "tests_passed": passed, "tests_total": len(cases), "cases": cases}
