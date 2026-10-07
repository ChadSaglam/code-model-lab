"""Tests for the lab: Task Lab levels, the local server's safety rules, own tasks and local models."""
from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
from http.server import ThreadingHTTPServer
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
lab = load("serve_lab")


# ---------------------------------------------------------------- Task Lab levels

def test_level_one_is_exactly_the_course_task_list():
    import glm53_flash.tasks as tasks
    course = [(f.name, f.arguments, f.descriptions, f.body, f.cases) for f in tasks.FAMILIES]
    assert course == [tuple(row) for row in ladder.LEVELS[1]["families"]]


def test_every_reference_solution_passes_its_own_tests():
    import glm53_flash.tasks as tasks
    from glm53_flash.evaluator import evaluate_source
    original = tasks.FAMILIES, tasks.FAMILY_BY_NAME
    try:
        ladder.activate(list(ladder.LEVELS))
        failures = [t.task_id for t in tasks.frozen_tasks("confirm", per_family=4)
                    if not evaluate_source(t, t.reference_source).passed]
    finally:
        tasks.FAMILIES, tasks.FAMILY_BY_NAME = original
    assert failures == []


def test_stop_rule_waits_for_the_final_return():
    import glm53_flash.tasks as tasks
    family = tasks.Family(*ladder.LEVELS[3]["families"][0])  # sign: several branches
    task = tasks.make_task(family, split="confirm", index=0, seed=1)
    assert not ladder.stops_at_function_end(task, "\n    if x > 0:\n        return 1\n")
    assert ladder.stops_at_function_end(task, task.reference_completion)


def test_context_fits_the_longest_example():
    size = ladder.lengths(list(ladder.LEVELS))
    longest = max(len(ladder.PROMPT.format(description=d, name=r[0], arguments=r[1]).encode()) + len(r[3].encode())
                  for level in ladder.LEVELS.values() for r in level["families"] for d in r[2])
    assert longest + 2 <= size["sequence_length"] + 1


# ---------------------------------------------------------------- settings validation

@pytest.mark.parametrize("raw, message", [
    ({"group_size": 1.5}, "whole number"),
    ({"rl_lr": 5}, "between"),
    ({"temperature": "hot"}, "must be a number"),
    ({"pretrain_steps": 500}, "one of"),
])
def test_bad_settings_are_rejected(raw, message):
    with pytest.raises(lab.LabError, match=message):
        lab.clean_settings(raw)


def test_course_defaults_are_kept_when_nothing_changes():
    assert lab.clean_settings({}) == lab.DEFAULT_SETTINGS


# ---------------------------------------------------------------- server safety rules

@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "RUNS", tmp_path)
    monkeypatch.setattr(lab, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(lab, "ARCHIVE", tmp_path / "_archive")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    port = httpd.server_address[1]
    monkeypatch.setattr(lab, "ALLOWED_HOSTS", {f"127.0.0.1:{port}"})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port
    httpd.shutdown()


def request(port: int, method: str, path: str, body: dict | None = None, headers: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode() if body is not None else None
    all_headers = {"Content-Type": "application/json"} if body is not None else {}
    all_headers.update(headers or {})
    connection.request(method, path, body=payload, headers=all_headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read() or b"{}")


def test_own_page_can_read_state(server):
    status, data = request(server, "GET", "/api/state?exp=baseline")
    assert status == 200 and data["experiment"]["id"] == "baseline"


def test_other_websites_are_blocked(server):
    status, _ = request(server, "POST", "/api/stop", {}, {"Origin": "http://evil.example"})
    assert status == 403


def test_dns_rebinding_host_is_blocked(server):
    status, _ = request(server, "GET", "/api/state", headers={"Host": f"evil.example:{server}"})
    assert status == 403


def test_actions_require_json(server):
    connection = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    connection.request("POST", "/api/stop", body=b"stage=rl", headers={"Content-Type": "text/plain"})
    assert connection.getresponse().status == 415


def test_unknown_stage_is_rejected(server):
    status, data = request(server, "POST", "/api/start", {"exp": "baseline", "stage": "rm -rf"})
    assert status == 400 and "Unknown stage" in data["error"]


def test_models_page_is_served(server):
    connection = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    connection.request("GET", "/models")
    response = connection.getresponse()
    assert response.status == 200 and b'id="tab-models"' in response.read()


def test_task_lab_run_needs_levels(server):
    status, data = request(server, "POST", "/api/experiments", {"name": "x", "kind": "ladder", "task_levels": []})
    assert status == 400 and "level" in data["error"]


# ---------------------------------------------------------------- answers from models and people

import subprocess  # noqa: E402

local_models = load("local_models")
text_tasks = load("text_tasks")
task_packs = load("task_packs")

FIXTURES = ROOT / "tests" / "fixtures"


def run_try(request: dict, env: dict | None = None) -> dict:
    import os
    done = subprocess.run([sys.executable, str(ROOT / "lab" / "try_task.py")], input=json.dumps(request),
                          capture_output=True, text=True, timeout=120, env={**os.environ, **(env or {})})
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_function_is_found_in_a_chat_style_answer():
    reply = "<think>plan</think>Sure!\n```python\ndef add_x(a, b):\n    return a + b\n```\nThis adds them."
    assert local_models.extract_function(reply, "add_x") == "def add_x(a, b):\n    return a + b\n"


def test_explanation_after_plain_code_is_dropped():
    reply = "def sign_x(x):\n    if x > 0:\n        return 1\n    return 0\nThat covers it."
    assert local_models.extract_function(reply, "sign_x").endswith("return 0\n")


def test_check_reports_every_case_and_friendly_errors():
    import glm53_flash.tasks as tasks
    task = tasks.make_task(tasks.FAMILY_BY_NAME["double"], split="confirm", index=0, seed=1)
    good = local_models.check(task, task.reference_source)
    assert good["passed"] and len(good["cases"]) == 3 and all(case["ok"] for case in good["cases"])
    looped = local_models.check(task, f"def {task.entry_point}(x):\n    for v in [x]:\n        return v * 2\n")
    assert looped["status"] == "invalid" and "loop" in looped["message"]


def test_own_task_is_validated_against_its_own_tests():
    ok = run_try({"mode": "validate", "description": "Return the larger absolute value of a and b.",
                  "code": "def bigger_abs(a, b):\n    if abs(a) > abs(b):\n        return abs(a)\n    return abs(b)\n",
                  "tests": "3, -7 -> 7\n-2, 1 -> 2\n0, 0 -> 0"})
    assert ok["ok"] and ok["task"]["arguments"] == "a, b" and len(ok["task"]["cases"]) == 3
    wrong = run_try({"mode": "validate", "description": "Return the larger absolute value.",
                     "code": "def bigger_abs(a, b):\n    return a\n", "tests": "3, -7 -> 7\n-2, 1 -> 2"})
    assert not wrong["ok"] and "doesn't pass" in wrong["error"]
    builtin = run_try({"mode": "validate", "description": "Clash with a built-in task.",
                       "code": "def sign(x):\n    return 1\n", "tests": "1 -> 1\n2 -> 1"})
    assert not builtin["ok"] and "built-in" in builtin["error"]


def test_wrong_function_name_is_reported_not_crashed():
    answer = run_try({"mode": "check", "family": "increment", "index": 0, "code": "def f(x):\n    return [0] * 10\n"})
    assert answer["ok"] and answer["status"] == "invalid" and "requested function" in answer["message"]


@pytest.fixture()
def fake_ollama():
    """A stand-in for Ollama that answers every task with `return <first input> + 1` in a markdown block,
    and can create and delete models like Ollama does."""
    import re
    from http.server import BaseHTTPRequestHandler

    models = {"fake:latest": {"family": "fake", "parameter_size": "1B", "quantization_level": "Q4_0"},
              "other:latest": {"family": "fake", "parameter_size": "2B", "quantization_level": "Q4_0"}}

    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": name, "size": 1000, "details": details} for name, details in models.items()]})

        def do_DELETE(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if models.pop(request["model"], None) is None:
                return self._send({"error": "model not found"}, 404)
            self._send({})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/create":
                models[request["model"]] = dict(models[request["from"]])
                return self._send({"status": "success"})
            if "think" in request:  # behave like a model without a thinking mode
                return self._send({"error": "model does not support thinking"}, 400)
            match = re.search(r"def (\w+)\(([^)]*)\):", request["prompt"])
            if match is None:  # a text task: parrot the prompt's last line as the answer
                line = request["prompt"].strip().splitlines()[-1]
                return self._send({"response": line, "prompt_eval_count": 40, "eval_count": 12})
            name, args = match.groups()
            self._send({"response": f"```python\ndef {name}({args}):\n    return {args.split(',')[0]} + 1\n```",
                        "prompt_eval_count": 40, "eval_count": 12})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_local_model_answer_is_tested_and_tokens_are_counted(fake_ollama):
    answer = run_try({"mode": "ollama", "family": "increment", "index": 0, "model": "fake:latest"}, {"OLLAMA_HOST": fake_ollama})
    assert answer["ok"] and answer["passed"] and answer["read"] == 40 and answer["written"] == 12


def test_local_model_test_set_writes_a_result_file(fake_ollama, tmp_path):
    import os
    output = tmp_path / "result.json"
    subprocess.run([sys.executable, str(ROOT / "lab" / "ollama_eval.py"), "--model", "fake:latest", "--set", "course",
                    "--output", str(output)], check=True, capture_output=True, timeout=120,
                   env={**os.environ, "OLLAMA_HOST": fake_ollama})
    result = json.loads(output.read_text())
    assert result["summary"]["tasks_total"] == 24
    assert result["summary"]["tasks_solved"] == 8  # only "increment" is answered correctly by the fake
    assert result["summary"]["read"] == 24 * 40 and result["summary"]["written"] == 24 * 12


# ---------------------------------------------------------------- text-answer tasks

def test_exact_match_normalises_whitespace():
    assert text_tasks.exact_match("hello   world", "  hello world  ")
    assert text_tasks.exact_match("a\nb\tc", "a b c")
    assert not text_tasks.exact_match("ja", "nein")


def test_json_fields_match_compares_named_fields_and_survives_malformed():
    assert text_tasks.json_fields_match({"total": 100, "vat": 7.7}, '{"total": 100, "vat": 7.7, "extra": 1}')
    assert not text_tasks.json_fields_match({"total": 100}, '{"total": 99}')
    assert not text_tasks.json_fields_match({"missing": 1}, '{"total": 100}')  # named field absent
    assert not text_tasks.json_fields_match({"total": 100}, "not json at all")  # malformed → wrong, not crash
    assert not text_tasks.json_fields_match({"total": 100}, "[1, 2, 3]")  # JSON but not an object


def test_numeric_match_within_tolerance_and_survives_non_numbers():
    assert text_tasks.numeric_match(100.0, "100.4", 0.5)
    assert text_tasks.numeric_match(100.0, "  99.6 ", 0.5)
    assert not text_tasks.numeric_match(100.0, "101", 0.5)
    assert not text_tasks.numeric_match(100.0, "abc", 0.5)  # non-numeric → wrong, not crash
    assert not text_tasks.numeric_match(100.0, "", 0.5)
    assert not text_tasks.numeric_match(100.0, "inf", 0.5)


def test_score_dispatches_by_check_type_and_rejects_unknown_kind():
    assert text_tasks.score(text_tasks.TextTask("e", "p", "exact", "ja"), "  ja ")["passed"]
    assert text_tasks.score(text_tasks.TextTask("j", "p", "json", {"a": 1}), '{"a": 1, "b": 2}')["passed"]
    assert text_tasks.score(text_tasks.TextTask("n", "p", "numeric", 5.0, 0.1), "5.05")["passed"]
    assert text_tasks.score(text_tasks.TextTask("n", "p", "numeric", 5.0, 0.1), "oops")["status"] == "failed"
    with pytest.raises(ValueError, match="one of"):
        text_tasks.TextTask("x", "p", "regex", "y")


def test_text_task_never_runs_the_answer_as_code():
    # an answer that looks like Python is compared as text, never executed or sandbox-validated
    task = text_tasks.TextTask("x", "p", "exact", "import os")
    assert text_tasks.score(task, "import os")["passed"]
    assert not text_tasks.score(task, "import sys")["passed"]


def test_text_task_is_scored_against_a_local_model(fake_ollama, monkeypatch):
    monkeypatch.setattr(local_models, "OLLAMA", fake_ollama)  # the fake parrots the prompt's last line
    good = local_models.answer_text_task("fake:latest",
                                         text_tasks.TextTask("num-1", "Reply with the total.\n42", "numeric", 42.0, 0.5))
    assert good["passed"] and good["read"] == 40 and good["written"] == 12
    wrong = local_models.answer_text_task("fake:latest",
                                          text_tasks.TextTask("num-2", "Reply with the total.\n7", "numeric", 42.0, 0.5))
    assert not wrong["passed"] and wrong["status"] == "failed"
    as_json = local_models.answer_text_task("fake:latest",
                                            text_tasks.TextTask("json-1", 'Reply with JSON.\n{"vat": 7.7}', "json", {"vat": 7.7}))
    assert as_json["passed"]


# ---------------------------------------------------------------- task packs

def _pack(**overrides) -> dict:
    pack = {"id": "p1", "name": "Pack one", "tasks": [
        {"id": "exact-1", "input": "Reply ja.", "expected": "ja", "check": "exact"},
        {"id": "json-1", "input": "Reply JSON.", "expected": {"vat": 7.7}, "check": "json"},
        {"id": "num-1", "input": "Reply the total.", "expected": 42.0, "check": "numeric", "tolerance": 0.5},
    ]}
    pack.update(overrides)
    return pack


def test_valid_pack_loads_into_text_tasks():
    pack = task_packs.validate_pack(_pack())
    assert pack.pack_id == "p1" and pack.name == "Pack one"
    assert [t.task_id for t in pack.tasks] == ["exact-1", "json-1", "num-1"]
    numeric = pack.tasks[2]
    assert numeric.check == "numeric" and numeric.tolerance == 0.5
    # the loaded tasks are the B-03 TextTask kind and score through the same rules
    assert text_tasks.score(pack.tasks[0], "  ja ")["passed"]


def test_example_fixture_pack_loads_from_disk():
    pack = task_packs.load_pack(FIXTURES / "example_pack.json")
    assert pack.pack_id == "example-1" and len(pack.tasks) == 3
    assert {t.check for t in pack.tasks} == {"exact", "json", "numeric"}


def test_missing_field_is_rejected_naming_task_and_field():
    tasks = _pack()["tasks"]
    del tasks[1]["expected"]
    with pytest.raises(ValueError, match="task 'json-1' is missing required field 'expected'"):
        task_packs.validate_pack(_pack(tasks=tasks))


def test_unknown_check_type_is_rejected():
    tasks = _pack()["tasks"]
    tasks[0]["check"] = "regex"
    with pytest.raises(ValueError, match="task 'exact-1': field 'check' must be one of"):
        task_packs.validate_pack(_pack(tasks=tasks))


def test_duplicate_task_id_is_rejected():
    tasks = _pack()["tasks"]
    tasks[1]["id"] = "exact-1"
    with pytest.raises(ValueError, match="task 'exact-1': duplicate task id"):
        task_packs.validate_pack(_pack(tasks=tasks))


def test_numeric_expected_must_be_a_number():
    tasks = _pack()["tasks"]
    tasks[2]["expected"] = "lots"
    with pytest.raises(ValueError, match="task 'num-1': field 'expected' must be a number"):
        task_packs.validate_pack(_pack(tasks=tasks))


def test_tolerance_only_applies_to_numeric_checks():
    tasks = _pack()["tasks"]
    tasks[0]["tolerance"] = 0.1
    with pytest.raises(ValueError, match="task 'exact-1': field 'tolerance' applies only to a numeric check"):
        task_packs.validate_pack(_pack(tasks=tasks))


def test_pack_without_tasks_is_rejected():
    with pytest.raises(ValueError, match="field 'tasks' must be a non-empty list"):
        task_packs.validate_pack(_pack(tasks=[]))


def test_pack_missing_id_is_rejected():
    data = _pack()
    del data["id"]
    with pytest.raises(ValueError, match="pack: field 'id' must be a non-empty string"):
        task_packs.validate_pack(data)


def test_load_pack_rejects_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        task_packs.load_pack(bad)


# ---------------------------------------------------------------- Models page

@pytest.fixture()
def models_server(server, fake_ollama, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LOCAL", tmp_path / "local-models")
    monkeypatch.setattr(lab, "VARIANTS", tmp_path / "local-models" / "variants.json")
    monkeypatch.setattr(lab.local_models, "OLLAMA", fake_ollama)
    monkeypatch.setenv("OLLAMA_HOST", fake_ollama)
    yield server
    lab.JOB.stop()


def test_variant_is_created_listed_and_only_lab_models_can_be_deleted(models_server):
    status, data = request(models_server, "POST", "/api/local-models/create", {"base": "fake:latest", "name": "Bad Name!", "system": "x" * 20})
    assert status == 400 and "lowercase" in data["error"]
    status, data = request(models_server, "POST", "/api/local-models/create",
                           {"base": "fake:latest", "name": "fake-rules", "system": "Never use loops."})
    assert status == 200 and data["model"] == "fake-rules:latest"
    _, info = request(models_server, "GET", "/api/local-models")
    made = {m["name"]: m["variant"] for m in info["details"]}
    assert made["fake-rules:latest"]["base"] == "fake:latest" and made["fake:latest"] is None
    status, _ = request(models_server, "POST", "/api/local-models/delete", {"name": "fake:latest"})
    assert status == 403  # a model the user installed is never deleted from the lab
    status, _ = request(models_server, "POST", "/api/local-models/delete", {"name": "fake-rules:latest"})
    assert status == 200
    _, info = request(models_server, "GET", "/api/local-models")
    assert "fake-rules:latest" not in info["models"] and "fake:latest" in info["models"]


def test_several_models_are_tested_one_after_another(models_server):
    import time
    status, _ = request(models_server, "POST", "/api/local-runs", {"models": ["fake:latest", "missing:latest"], "set": "course"})
    assert status == 400  # every model must be installed
    status, _ = request(models_server, "POST", "/api/local-runs", {"models": ["fake:latest", "other:latest"], "set": "course"})
    assert status == 200
    _, data = request(models_server, "GET", "/api/local-runs")
    assert data["queued"] == ["other:latest"]
    deadline = time.time() + 120
    while time.time() < deadline:
        _, data = request(models_server, "GET", "/api/local-runs")
        if not data["queued"] and all(run["status"] == "done" for run in data["runs"]) and len(data["runs"]) == 2:
            break
        time.sleep(0.3)
    assert sorted(run["model"] for run in data["runs"]) == ["fake:latest", "other:latest"]
    assert all(run["summary"]["tasks_solved"] == 8 for run in data["runs"])
