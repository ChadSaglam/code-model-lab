#!/usr/bin/env python3
"""Code Model Lab: train the small model, test it on harder tasks and compare it with local Ollama models.

Start it from the project folder:
    .venv/bin/python lab/serve_lab.py
It opens http://127.0.0.1:8765 in your browser (Model Lab; Task Lab at /tasks, Models at /models).
Stop it with Ctrl + C.

Every experiment is one full run of the course (pretrain -> RL -> two tests) with its own settings.
The run you made from the README is shown as the "Course baseline" experiment; its files are not moved.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder  # noqa: E402  (lab/ladder.py: Task Lab difficulty levels)
import local_models  # noqa: E402  (lab/local_models.py: Ollama client and safe test runs)

HOST, PORT = "127.0.0.1", 8765
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
PAGE = Path(__file__).resolve().parent / "index.html"
RUNS = ROOT / "runs"
EXPERIMENTS = RUNS / "experiments"
ARCHIVE = RUNS / "_archive"
EXTERNAL_RUN_GRACE_SECONDS = 180  # a run started in Terminal counts as "still active" this long after its last write

BASELINE = "baseline"
RL_FAMILIES = "increment,double,even"
FAMILIES = ["increment", "double", "even", "square", "absolute", "nonnegative", "reverse", "list_sum"]
STAGE_ORDER = ["pretrain", "rl", "eval_before", "eval_after"]
DOWNSTREAM = {"pretrain": ["rl", "eval_before", "eval_after"], "rl": ["eval_after"], "eval_before": [], "eval_after": []}

# The course's settings. An experiment changes some of them; everything else stays fixed so runs stay comparable.
DEFAULT_SETTINGS = {"seed": 42, "pretrain_lr": 3e-4, "rl_seed": 31415, "rl_lr": 5e-5, "temperature": 0.35,
                    "group_size": 16, "pretrain_steps": 400, "rl_groups": 96}
SETTING_LIMITS = {
    "seed": (int, 0, 1_000_000),
    "pretrain_lr": (float, 1e-6, 1e-2),
    "rl_seed": (int, 0, 1_000_000),
    "rl_lr": (float, 1e-7, 1e-2),
    "temperature": (float, 0.05, 2.0),
    "group_size": (int, 2, 32),
}
SETTING_CHOICES = {"pretrain_steps": (400, 800, 1200), "rl_groups": (96, 192)}  # Task Lab only; the course keeps 400 / 96


class LabError(Exception):
    """A problem to show on the page, with its HTTP status."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- experiments

@dataclass
class Paths:
    pretrain: Path
    rl: Path
    eval_before: Path
    eval_after: Path
    logs: Path


def baseline_record() -> dict:
    return {"id": BASELINE, "name": "Course baseline", "created": None, "settings": dict(DEFAULT_SETTINGS),
            "pretrain_from": None, "task_levels": None, "color_slot": 0, "environment": None}


def kind_of(record: dict) -> str:
    return "ladder" if record.get("task_levels") else "course"


_JSON_CACHE: dict[str, tuple[int, dict]] = {}


def read_json(path: Path) -> dict | None:
    """Parsed JSON, cached until the file changes (RL receipts are several MB and are read on every refresh)."""
    try:
        stamp = path.stat().st_mtime_ns
        cached = _JSON_CACHE.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        data = json.loads(path.read_text())
        _JSON_CACHE[str(path)] = (stamp, data)
        return data
    except (OSError, json.JSONDecodeError):
        return None


def all_experiments(kind: str | None = None) -> list[dict]:
    records = [baseline_record()]
    for file in sorted(EXPERIMENTS.glob("*/experiment.json")):
        record = read_json(file)
        if record and record.get("id") == file.parent.name:
            record.setdefault("task_levels", None)
            records.append(record)
    records = [r for r in records if kind is None or kind_of(r) == kind]
    return sorted(records, key=lambda r: (r["id"] != BASELINE, r.get("created") or ""))


def experiment(exp_id: str | None) -> dict:
    for record in all_experiments():
        if record["id"] == (exp_id or BASELINE):
            return record
    raise LabError("Unknown experiment.", 404)


def paths_for(record: dict) -> Paths:
    if record["id"] == BASELINE:  # the run made from the README, at its original paths
        paths = Paths(RUNS / "glm53-coding-pretrain-001", RUNS / "glm53-executable-rloo-diverse-001",
                      RUNS / "confirm-greedy-pretrain-0100.json", RUNS / "confirm-greedy-rl-0096.json",
                      RUNS / "_lab_logs")
    else:
        base = EXPERIMENTS / record["id"]
        paths = Paths(base / "pretrain", base / "rl", base / "eval-before.json", base / "eval-after.json", base / "_logs")
    if record.get("pretrain_from"):  # reused pretraining: stage 1 and the "before" test belong to the source
        source = paths_for(experiment(record["pretrain_from"]))
        paths.pretrain, paths.eval_before = source.pretrain, source.eval_before
    return paths


def plan(record: dict) -> dict:
    """The numbers that shape a run. The course is fixed; Task Lab sizes context and answers to its levels."""
    s = record["settings"]
    if kind_of(record) == "course":
        return {"steps": 400, "rl_start": 100, "groups": 96, "sequence_length": 128, "max_new_tokens": 48,
                "tests": 24, "eval": ["--per-family", "8", "--families", RL_FAMILIES],
                "rl_tasks": ["--tasks-per-family", "32", "--families", RL_FAMILIES]}
    size = ladder.lengths(record["task_levels"])
    steps = s.get("pretrain_steps", 400)
    return {"steps": steps, "rl_start": steps, "groups": s.get("rl_groups", 96),
            "sequence_length": size["sequence_length"], "max_new_tokens": size["max_new_tokens"],
            "tests": size["families"] * 4, "eval": ["--per-family", "4"], "rl_tasks": ["--tasks-per-family", "8"]}


def stage_specs(record: dict) -> "OrderedDict[str, dict]":
    s, p, n = record["settings"], paths_for(record), plan(record)
    start = p.pretrain / f"checkpoint-{n['rl_start']:04d}"
    final = p.rl / f"checkpoint-{n['groups']:04d}"
    shared = record.get("pretrain_from")
    tokens = ["--max-new-tokens", str(n["max_new_tokens"])]
    # Task Lab runs the unchanged course scripts through lab/ladder.py, which swaps in the chosen task levels.
    runner = [] if kind_of(record) == "course" else ["lab/ladder.py", "--levels", ",".join(map(str, record["task_levels"]))]

    def spec(output: Path, done: Path, needs: list[Path], args: list[str], total: int, shared_from: str | None = None) -> dict:
        return {"output": output, "done": done, "needs": needs, "args": runner + args, "total": total, "shared_from": shared_from}

    def evaluate(checkpoint: Path, output: Path) -> list[str]:
        return ["scripts/evaluate.py", "--checkpoint", str(checkpoint), "--split", "confirm", *n["eval"], *tokens,
                "--output", str(output), "--device", "auto"]

    marks = sorted({100, n["steps"] // 2, n["steps"]})
    rounds = [n["groups"] * i // 4 for i in range(1, 5)]
    return OrderedDict([
        ("pretrain", spec(p.pretrain, p.pretrain / "training-receipt.json", [], [
            "scripts/train_pretrain.py", "--output", str(p.pretrain), "--steps", str(n["steps"]),
            "--checkpoints", ",".join(map(str, marks)), "--batch-size", "32",
            "--sequence-length", str(n["sequence_length"]), "--learning-rate", str(s["pretrain_lr"]),
            "--seed", str(s["seed"]), "--device", "auto"], n["steps"], shared)),
        ("rl", spec(p.rl, p.rl / "training-receipt.json", [start / "model.pt"], [
            "scripts/train_rl.py", "--initial-checkpoint", str(start), "--output", str(p.rl),
            "--groups", str(n["groups"]), "--checkpoints", ",".join(map(str, rounds)), "--group-size", str(s["group_size"]),
            *n["rl_tasks"], *tokens, "--temperature", str(s["temperature"]),
            "--learning-rate", str(s["rl_lr"]), "--seed", str(s["rl_seed"]),
            "--train-scope", "last-block-head", "--reward-mode", "binary", "--device", "auto"], n["groups"])),
        ("eval_before", spec(p.eval_before, p.eval_before, [start / "model.pt"], evaluate(start, p.eval_before), n["tests"], shared)),
        ("eval_after", spec(p.eval_after, p.eval_after, [final / "model.pt"], evaluate(final, p.eval_after), n["tests"])),
    ])


def snapshots_for(record: dict) -> "OrderedDict[str, tuple[str, Path]]":
    p = paths_for(record)
    return OrderedDict([
        ("pre-0000", ("Random start", p.pretrain / "checkpoint-0000")),
        ("pre-0100", ("Pretraining · step 100", p.pretrain / "checkpoint-0100")),
        ("pre-0200", ("Pretraining · step 200", p.pretrain / "checkpoint-0200")),
        ("pre-0400", ("Pretraining · step 400", p.pretrain / "checkpoint-0400")),
        ("rl-0024", ("RL · round 24", p.rl / "checkpoint-0024")),
        ("rl-0048", ("RL · round 48", p.rl / "checkpoint-0048")),
        ("rl-0072", ("RL · round 72", p.rl / "checkpoint-0072")),
        ("rl-0096", ("RL · round 96 (final)", p.rl / "checkpoint-0096")),
    ])


def clean_settings(raw: dict) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    for key, (kind, low, high) in SETTING_LIMITS.items():
        if key not in raw or raw[key] in ("", None):
            continue
        try:
            value = kind(raw[key])
        except (TypeError, ValueError):
            raise LabError(f"'{key}' must be a number.", 400) from None
        if kind is int and value != float(raw[key]):
            raise LabError(f"'{key}' must be a whole number.", 400)
        if not low <= value <= high:
            raise LabError(f"'{key}' must be between {low:g} and {high:g}.", 400)
        settings[key] = value
    for key, choices in SETTING_CHOICES.items():
        if raw.get(key) not in ("", None):
            if raw[key] not in choices:
                raise LabError(f"'{key}' must be one of {', '.join(map(str, choices))}.", 400)
            settings[key] = raw[key]
    return settings


def clean_levels(raw) -> list[int] | None:
    if raw in (None, [], ""):
        return None
    if not isinstance(raw, list) or not all(isinstance(v, int) and v in ladder.usable_levels() for v in raw):
        raise LabError("Pick at least one difficulty level.", 400)
    return sorted(set(raw))


def environment() -> dict:
    import torch  # imported here so the page itself starts instantly
    return {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine(),
            "system": platform.system()}


def create_experiment(data: dict) -> dict:
    name = str(data.get("name", "")).strip()
    if not 1 <= len(name) <= 40:
        raise LabError("Give the experiment a name (1–40 characters).", 400)
    levels = clean_levels(data.get("task_levels"))
    if data.get("kind") == "ladder" and not levels:
        raise LabError("Pick at least one difficulty level.", 400)
    settings = clean_settings(data.get("settings") or {})
    if not levels:
        settings["pretrain_steps"], settings["rl_groups"] = 400, 96  # the course run is fixed
    pretrain_from = data.get("pretrain_from") or None
    if pretrain_from:
        source = experiment(pretrain_from)
        if source.get("pretrain_from"):
            source = experiment(source["pretrain_from"])  # always point at the experiment that owns the pretraining
        if source.get("task_levels") != levels or source["settings"].get("pretrain_steps", 400) != settings["pretrain_steps"]:
            raise LabError("Pretraining can only be reused from a run with the same levels and steps.", 400)
        pretrain_from = source["id"]
        settings["seed"], settings["pretrain_lr"] = source["settings"]["seed"], source["settings"]["pretrain_lr"]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "experiment"
    exp_id = f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    # A fixed color slot per experiment, so archiving one never repaints the others.
    ever_created = len(list(EXPERIMENTS.glob("*/experiment.json"))) + len(list(ARCHIVE.glob("experiment-*")))
    record = {"id": exp_id, "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": settings,
              "pretrain_from": pretrain_from, "task_levels": levels, "color_slot": (ever_created + 1) % 8,
              "environment": environment()}
    if levels:
        record["plan"] = ladder.lengths(levels)
    folder = EXPERIMENTS / exp_id
    folder.mkdir(parents=True)
    (folder / "experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


# ---------------------------------------------------------------- running stages

class Job:
    """The one training or evaluation process this server runs, plus the stages queued after it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.exp: str | None = None
        self.stage: str | None = None
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.started = 0.0
        self.returncode: int | None = None
        self.errors: dict[tuple[str, str], list[str]] = {}
        self.queue: list[str] = []
        self.queue_exp: str | None = None
        self.local_queue: list[dict] = []  # local-model tests waiting to run, one model after another

    def running(self, exp: str | None = None, stage: str | None = None) -> bool:
        active = self.proc is not None and self.proc.poll() is None
        return active and (exp is None or exp == self.exp) and (stage is None or stage == self.stage)

    def start(self, record: dict, stage: str) -> None:
        self.run_command(record["id"], stage, stage_specs(record)[stage]["args"], paths_for(record).logs)

    def run_command(self, owner: str, stage: str, args: list[str], logs: Path) -> None:
        """Start one lab script; owner is an experiment id, or "local:<id>" for a local-model test."""
        logs.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, *args], cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.exp, self.stage, self.lines, self.started, self.returncode = owner, stage, [], time.time(), None
        self.errors.pop((owner, stage), None)
        threading.Thread(target=self._collect_output, args=(self.proc, owner, stage, logs), daemon=True).start()

    def _collect_output(self, proc: subprocess.Popen, exp: str, stage: str, logs: Path) -> None:
        assert proc.stdout is not None
        with (logs / f"{stage}.log").open("w") as log:
            for line in proc.stdout:
                line = line.rstrip("\n")
                with self.lock:
                    self.lines.append(line)
                log.write(line + "\n")
                log.flush()
        code = proc.wait()
        with self.lock:
            self.returncode = code
            if code != 0:
                self.errors[(exp, stage)] = self.lines[-25:]
                self.queue = []  # a failed training stage stops its run; queued model tests still go ahead
        start_next_queued()

    def stop(self) -> None:
        self.queue = []
        self.local_queue = []
        if self.running():
            assert self.proc is not None
            self.proc.terminate()


JOB = Job()
CONTROL_LOCK = threading.Lock()  # makes "check, then start/archive" one step, so two clicks can't start two runs


def json_lines(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def stage_lines(record: dict, stage: str) -> list[str]:
    """Live output if this server runs the stage, else the log of its last run here."""
    with JOB.lock:
        if JOB.exp == record["id"] and JOB.stage == stage:
            return list(JOB.lines)
    path = paths_for(record).logs / f"{stage}.log"
    return path.read_text().splitlines() if path.exists() else []


def live_progress(record: dict, stage: str) -> int:
    rows = json_lines(stage_lines(record, stage))
    if stage == "pretrain":
        return max((r["step"] for r in rows if "step" in r), default=0)
    if stage == "rl":
        return max((r["group"] for r in rows if "group" in r), default=0)
    return sum(1 for r in rows if "task" in r)


def seconds_since_change(path: Path) -> float:
    times = [path.stat().st_mtime]
    if path.is_dir():
        times += [child.stat().st_mtime for child in path.iterdir()]
    return time.time() - max(times)


def saved_checkpoint_progress(output: Path) -> int:
    numbers = [p.name.split("-")[1] for p in output.glob("checkpoint-*")]
    return max((int(n) for n in numbers if n.isdigit()), default=0)


def stage_state(record: dict, stage: str) -> dict:
    spec = stage_specs(record)[stage]
    state = {"progress": 0, "total": spec["total"], "shared_from": spec["shared_from"]}
    if spec["shared_from"]:
        owner = experiment(spec["shared_from"])
        return {**stage_state(owner, stage), "shared_from": owner["id"], "shared_name": owner["name"]}
    if JOB.running(record["id"], stage):
        return {**state, "status": "running", "progress": live_progress(record, stage), "since": JOB.started}
    if spec["done"].exists():
        return {**state, "status": "done", "progress": spec["total"], **stage_timing(spec["done"])}
    if (record["id"], stage) in JOB.errors:
        return {**state, "status": "failed", "error": JOB.errors[(record["id"], stage)]}
    if spec["output"].exists():
        progress = saved_checkpoint_progress(spec["output"]) if spec["output"].is_dir() else 0
        return {**state, "status": "outside", "progress": progress,
                "seconds_since_change": round(seconds_since_change(spec["output"]))}
    if all(path.exists() for path in spec["needs"]):
        return {**state, "status": "ready"}
    return {**state, "status": "locked"}


def stage_timing(done: Path) -> dict:
    """How long a finished stage took, when it finished, and how many tokens it read and wrote.

    The small model reads and writes one token per byte. Pretraining reads its training text;
    RL and the tests read their prompts and write answers.
    """
    receipt = read_json(done) or {}
    tokens = {"read": None, "written": None}
    if "training_curve" in receipt:
        tokens["read"] = receipt.get("tokens_seen")
    elif "groups_detail" in receipt:
        tokens["written"] = sum(sample.get("tokens", 0) for group in receipt["groups_detail"] for sample in group.get("samples", []))
    elif "episodes" in receipt:
        tokens["read"] = sum(len(e["prompt"].encode()) + 1 for e in receipt["episodes"])
        tokens["written"] = sum(e.get("generated_tokens", 0) for e in receipt["episodes"])
    return {"seconds": receipt.get("elapsed_seconds"), "finished": done.stat().st_mtime, "tokens": tokens}


def run_seconds(record: dict) -> float | None:
    """Total training and test time of this run's own stages (reused stages are counted where they ran)."""
    states = [stage_state(record, s) for s in STAGE_ORDER]
    own = [st for st in states if not st.get("shared_from") and st["status"] == "done" and st.get("seconds")]
    return round(sum(st["seconds"] for st in own), 1) if own else None


def check_startable(record: dict, stage: str) -> None:
    state = stage_state(record, stage)
    if state.get("shared_from"):
        raise LabError(f"This stage is reused from '{state['shared_name']}'. Run it there.")
    if state["status"] == "locked":
        raise LabError("The previous stage has to finish first.")
    if state["status"] in ("done", "outside") or stage_specs(record)[stage]["output"].exists():
        raise LabError("This stage already has a result. Click 'Archive and redo' first.")


def start_stage(record: dict, stage: str) -> None:
    with CONTROL_LOCK:
        if JOB.running():
            raise LabError("Another stage is running. Wait for it to finish first.")
        check_startable(record, stage)
        JOB.queue = []
        JOB.start(record, stage)


def run_all(record: dict) -> list[str]:
    """Start the first unfinished stage and queue the rest; each one starts when the previous succeeds."""
    with CONTROL_LOCK:
        if JOB.running():
            raise LabError("Another stage is running. Wait for it to finish first.")
        todo = [s for s in STAGE_ORDER if not stage_state(record, s).get("shared_from")
                and stage_state(record, s)["status"] in ("ready", "locked")]
        if not todo:
            raise LabError("Nothing left to run in this experiment.")
        check_startable(record, todo[0])
        JOB.queue, JOB.queue_exp = todo[1:], record["id"]
        JOB.start(record, todo[0])
        return todo


def start_next_queued() -> None:
    with CONTROL_LOCK:
        if JOB.running():
            return
        if not JOB.queue:
            if JOB.local_queue:
                launch_local_run(JOB.local_queue.pop(0))
            return
        stage = JOB.queue.pop(0)
        try:
            record = experiment(JOB.queue_exp)
            check_startable(record, stage)
        except LabError:
            JOB.queue = []
            return
        JOB.start(record, stage)


def users_of_pretraining(owner_id: str) -> list[str]:
    return [r["name"] for r in all_experiments() if r.get("pretrain_from") == owner_id]


def archive_target(path: Path, stamp: str, exp_id: str) -> Path:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    return ARCHIVE / f"{exp_id}-{path.stem}-{stamp}{path.suffix}"


def archive_stage(record: dict, stage: str) -> list[str]:
    """Move this stage's result, and every result built on it, into runs/_archive. Nothing is deleted."""
    with CONTROL_LOCK:
        if stage_specs(record)[stage]["shared_from"]:
            raise LabError("This stage is reused from another experiment. Archive it there.")
        affected = [s for s in [stage, *DOWNSTREAM[stage]] if not stage_specs(record)[s]["shared_from"]]
        if stage == "pretrain" and users_of_pretraining(record["id"]):
            raise LabError(f"Other experiments reuse this pretraining: {', '.join(users_of_pretraining(record['id']))}.")
        if any(JOB.running(record["id"], s) for s in affected):
            raise LabError("A stage that uses this result is running. Stop it first.")
        for each in affected:
            state = stage_state(record, each)
            if state["status"] == "outside" and state["seconds_since_change"] < EXTERNAL_RUN_GRACE_SECONDS:
                raise LabError("A run in Terminal still seems to be writing here. Wait until it finishes.")
        stamp, specs, logs = time.strftime("%Y%m%d-%H%M%S"), stage_specs(record), paths_for(record).logs
        moved = []
        for each in affected:
            output = specs[each]["output"]
            if output.exists():
                shutil.move(str(output), str(archive_target(output, stamp, record["id"])))
                moved.append(each)
            log = logs / f"{each}.log"
            if log.exists():
                shutil.move(str(log), str(archive_target(log, stamp, record["id"])))
            JOB.errors.pop((record["id"], each), None)
        COMPARE.forget_models()
        return moved


def archive_experiment(record: dict) -> None:
    with CONTROL_LOCK:
        if record["id"] == BASELINE:
            raise LabError("The course baseline can't be archived as a whole. Archive its stages instead.")
        if JOB.running(record["id"]):
            raise LabError("This experiment is running. Stop it first.")
        if users_of_pretraining(record["id"]):
            raise LabError(f"Other experiments reuse its pretraining: {', '.join(users_of_pretraining(record['id']))}.")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.move(str(EXPERIMENTS / record["id"]), str(ARCHIVE / f"experiment-{record['id']}"))
        COMPARE.forget_models()


# ---------------------------------------------------------------- reading results

def pretrain_curve(record: dict) -> dict:
    spec = stage_specs(record)["pretrain"]
    receipt = read_json(spec["done"])
    if receipt:
        return {"points": [[r["step"], round(r["loss"], 4)] for r in receipt.get("training_curve", [])],
                "complete": True, "seconds": receipt.get("elapsed_seconds"), "tokens": receipt.get("tokens_seen"),
                "device": receipt.get("device"), "parameters": receipt.get("parameter_counts")}
    owner = experiment(spec["shared_from"]) if spec["shared_from"] else record
    rows = json_lines(stage_lines(owner, "pretrain"))
    return {"points": [[r["step"], round(r["loss"], 4)] for r in rows if "step" in r and "loss" in r], "complete": False}


def rl_row(group: int, task_id: str, family: str, exact: int, size: int) -> dict:
    return {"group": group, "task": task_id, "family": family, "exact": exact, "size": size}


def rl_groups(record: dict) -> dict:
    receipt = read_json(stage_specs(record)["rl"]["done"])
    if receipt:
        rows = [rl_row(r["group"], r["task_id"], r["family"], r["exact_rollouts"], len(r["rewards"]))
                for r in receipt.get("groups_detail", [])]
        return {"groups": rows, "complete": True, "seconds": receipt.get("elapsed_seconds"),
                "trainable_parameters": receipt.get("trainable_parameters")}
    rows = [rl_row(r["group"], r["task_id"], r["task_id"].split("-")[1], r["exact_rollouts"], len(r["rewards"]))
            for r in json_lines(stage_lines(record, "rl")) if "group" in r and "task_id" in r]
    return {"groups": rows, "complete": False}


def eval_side(path: Path) -> dict | None:
    data = read_json(path)
    if not data:
        return None
    episodes = [{"task": e["task_id"], "family": e["family"], "prompt": e["prompt"],
                 "output": e["generated_completion"], "passed": e["evaluation"]["passed"],
                 "status": e["evaluation"]["status"],
                 "tests_passed": e["evaluation"]["tests_passed"], "tests_total": e["evaluation"]["tests_total"]}
                for e in data.get("episodes", [])]
    return {"summary": data.get("summary"), "by_family": data.get("by_family"), "episodes": episodes}


def eval_both(record: dict) -> dict:
    specs = stage_specs(record)
    return {"before": eval_side(specs["eval_before"]["done"]), "after": eval_side(specs["eval_after"]["done"])}


def rolling_rate(groups: list[dict], window: int = 12) -> list[list[float]]:
    series = []
    for i, g in enumerate(groups):
        recent = groups[max(0, i - window + 1): i + 1]
        series.append([g["group"], round(100 * sum(r["exact"] for r in recent) / sum(r["size"] for r in recent), 1)])
    return series


def rate(groups: list[dict]) -> float | None:
    attempts = sum(g["size"] for g in groups)
    return round(100 * sum(g["exact"] for g in groups) / attempts, 1) if attempts else None


def overview(kind: str) -> dict:
    """One row per experiment of this kind (course or ladder) with settings, key results and curves."""
    rows = []
    for record in all_experiments(kind):
        pretrain, rl, tests = pretrain_curve(record), rl_groups(record), eval_both(record)
        points, groups = pretrain["points"], rl["groups"]
        rows.append({
            "id": record["id"], "name": record["name"], "created": record.get("created"),
            "settings": record["settings"], "pretrain_from": record.get("pretrain_from"),
            "task_levels": record.get("task_levels"),
            "color_slot": record.get("color_slot", 0),
            "environment": record.get("environment"),
            "stages": {s: stage_state(record, s)["status"] for s in STAGE_ORDER},
            "final_loss": round(sum(p[1] for p in points[-10:]) / len(points[-10:]), 3) if pretrain["complete"] and points else None,
            "pretrain_seconds": pretrain.get("seconds"),
            "rl_first24": rate(groups[:24]) if len(groups) >= 24 else None,
            "rl_last24": rate(groups[-24:]) if rl["complete"] else None,
            "rl_seconds": rl.get("seconds"),
            "run_seconds": run_seconds(record),
            "before": tests["before"]["summary"]["tasks_solved"] if tests["before"] else None,
            "after": tests["after"]["summary"]["tasks_solved"] if tests["after"] else None,
            "tasks": plan(record)["tests"],
            "pretrain_curve": [] if record.get("pretrain_from") else points[::5] + points[-1:],
            "rl_curve": rolling_rate(groups),
        })
    return {"experiments": rows, "defaults": DEFAULT_SETTINGS, "choices": SETTING_CHOICES,
            "limits": {k: [lo, hi] for k, (_, lo, hi) in SETTING_LIMITS.items()}}


def level_results(record: dict) -> dict:
    """Task Lab: how much of each difficulty level was solved before and after RL, and during RL."""
    level_of, titles = ladder.family_levels(), {lv: info["title"] for lv, info in ladder.all_levels().items()}
    rows: dict[int, dict] = {}

    def row(level: int) -> dict:
        return rows.setdefault(level, {"level": level, "title": titles[level], "before_solved": 0, "before_total": 0,
                                       "after_solved": 0, "after_total": 0, "rl_correct": 0, "rl_attempts": 0})

    for side, data in eval_both(record).items():
        for episode in (data or {}).get("episodes", []):
            target = row(level_of.get(episode["family"], ladder.CUSTOM_LEVEL))
            target[f"{side}_solved"] += int(episode["passed"])
            target[f"{side}_total"] += 1
    for group in rl_groups(record)["groups"]:
        target = row(level_of.get(group["family"], ladder.CUSTOM_LEVEL))
        target["rl_correct"] += group["exact"]
        target["rl_attempts"] += group["size"]
    for level in record.get("task_levels") or [1]:
        row(level)
    return {"levels": [rows[level] for level in sorted(rows)]}


def ladder_info() -> dict:
    custom = [{"name": t["name"], "description": t["descriptions"][0], "tests": len(t["cases"])}
              for t in read_json_list(ladder.CUSTOM_FILE)]
    return {"levels": ladder.describe(), "family_levels": ladder.family_levels(), "custom": custom,
            "sizes": {",".join(map(str, combo)): ladder.lengths(list(combo))
                      for size in range(1, len(ladder.usable_levels()) + 1) for combo in combinations(ladder.usable_levels(), size)}}


class Compare:
    """Loads saved snapshots on the CPU and lets each one answer the same task."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.models: "OrderedDict[str, object]" = OrderedDict()

    def forget_models(self) -> None:
        with self.lock:
            self.models.clear()

    def _model(self, path: Path):
        import torch  # imported here so the page itself starts instantly
        from glm53_flash.runtime import load_checkpoint
        key = str(path)
        if key not in self.models:
            self.models[key] = load_checkpoint(path, torch.device("cpu")).eval()
            while len(self.models) > 4:  # each snapshot is ~100 MB in memory
                self.models.popitem(last=False)
        self.models.move_to_end(key)
        return self.models[key]

    def run(self, record: dict, family: str, index: int, keys: list[str]) -> dict:
        from glm53_flash import ByteTokenizer
        from glm53_flash.evaluator import evaluate_source
        from glm53_flash.runtime import generate_group
        from glm53_flash.tasks import frozen_tasks
        task = [t for t in frozen_tasks("confirm", per_family=8) if t.family == family][index]
        snapshots, tokenizer, answers = snapshots_for(record), ByteTokenizer(), []
        with self.lock:
            for key in keys:
                label, path = snapshots[key]
                started = time.perf_counter()
                generated = generate_group(self._model(path), tokenizer, task, group_size=1, max_new_tokens=48,
                                           temperature=1.0, sample=False, seed=2026 + index)[0]
                evaluation = evaluate_source(task, task.prompt + generated["completion"]).to_dict()
                answers.append({"key": key, "label": label, "output": generated["completion"],
                                "passed": evaluation["passed"], "status": evaluation["status"],
                                "tests_passed": evaluation["tests_passed"], "tests_total": evaluation["tests_total"],
                                "seconds": round(time.perf_counter() - started, 1)})
        return {"task": task.task_id, "prompt": task.prompt, "reference": task.reference_completion, "answers": answers}


COMPARE = Compare()


def lab_job() -> dict:
    with JOB.lock:
        job = {"exp": JOB.exp, "stage": JOB.stage, "running": JOB.running(), "returncode": JOB.returncode,
               "queue": list(JOB.queue), "log": JOB.lines[-40:]}
    if JOB.exp:
        job["exp_name"] = (f"Local model test · {local_record(JOB.exp[6:]).get('model', '')}" if JOB.exp.startswith("local:")
                           else next((r["name"] for r in all_experiments() if r["id"] == JOB.exp), JOB.exp))
    return job


def lab_state(record: dict) -> dict:
    with JOB.lock:
        job = {"exp": JOB.exp, "stage": JOB.stage, "running": JOB.running(), "returncode": JOB.returncode,
               "queue": list(JOB.queue), "log": JOB.lines[-40:]}
    if JOB.exp:
        job["exp_name"] = (f"Local model test · {local_record(JOB.exp[6:]).get('model', '')}" if JOB.exp.startswith("local:")
                           else next((r["name"] for r in all_experiments() if r["id"] == JOB.exp), JOB.exp))
    snapshots = [{"key": k, "label": label, "available": (path / "model.pt").exists()}
                 for k, (label, path) in snapshots_for(record).items()]
    return {"experiment": record, "stages": {s: stage_state(record, s) for s in STAGE_ORDER}, "job": job,
            "snapshots": snapshots, "families": FAMILIES}



# ---------------------------------------------------------------- local models, trying tasks, level 5

LOCAL = RUNS / "local-models"
TRY_TIMEOUTS = {"show": 60, "check": 60, "validate": 60, "snapshot": 240, "ollama": 330}


def local_record(run_id: str) -> dict:
    return read_json(LOCAL / run_id / "run.json") or {}


def local_runs() -> list[dict]:
    rows = []
    for file in sorted(LOCAL.glob("*/run.json"), reverse=True):
        record, folder = read_json(file) or {}, file.parent
        result = read_json(folder / "result.json")
        owner = f"local:{record.get('id')}"
        if JOB.running(owner):
            status, progress = "running", sum(1 for r in json_lines(list(JOB.lines)) if "task" in r)
        elif result:
            status, progress = "done", record.get("total", 0)
        elif (owner, "test") in JOB.errors:
            status, progress = "failed", 0
        else:
            status, progress = "stopped", 0
        by_level: dict[int, list[int]] = {}
        for episode in (result or {}).get("episodes", []):
            counts = by_level.setdefault(episode.get("level", 1), [0, 0])
            counts[0] += int(episode["passed"])
            counts[1] += 1
        rows.append({**record, "status": status, "progress": progress, "summary": (result or {}).get("summary"),
                     "by_level": {str(level): counts for level, counts in sorted(by_level.items())},
                     "error": JOB.errors.get((owner, "test"))})
    return rows


VARIANTS = LOCAL / "variants.json"  # models made on the Models page; only these can be deleted from the lab


def lab_variants() -> dict:
    data = read_json(VARIANTS)
    return data if isinstance(data, dict) else {}


def local_models_info() -> dict:
    variants = lab_variants()
    try:
        models = local_models.model_details()
    except local_models.ModelError as error:
        return {"available": False, "models": [], "details": [], "error": str(error), "host": local_models.OLLAMA}
    for model in models:
        model["variant"] = variants.get(model["name"])
    return {"available": True, "models": [m["name"] for m in models], "details": models, "host": local_models.OLLAMA}


def launch_local_run(request: dict) -> dict:
    """Start one local-model test now (the caller holds CONTROL_LOCK and nothing is running)."""
    model, test, levels = request["model"], request["set"], request["levels"]
    run_id = f"{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}-{time.strftime('%Y%m%d-%H%M%S')}"
    folder = LOCAL / run_id
    folder.mkdir(parents=True, exist_ok=True)
    total = 24 if test == "course" else ladder.lengths(levels)["families"] * 4
    record = {"id": run_id, "model": model, "set": test, "levels": levels, "total": total,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    args = ["lab/ollama_eval.py", "--model", model, "--set", test, "--output", str(folder / "result.json")]
    if levels:
        args += ["--levels", ",".join(map(str, levels))]
    JOB.run_command(f"local:{run_id}", "test", args, folder / "_logs")
    return record


def start_local_run(data: dict) -> dict:
    """Test one or several models; several run one after another."""
    test = data.get("set")
    models = data.get("models") or ([data["model"]] if data.get("model") else [])
    models = list(dict.fromkeys(models)) if isinstance(models, list) else []
    if test not in ("course", "levels"):
        raise LabError("Pick a test set.", 400)
    available = local_models_info()["models"]
    if not models or not all(isinstance(m, str) and m in available for m in models):
        raise LabError("Pick at least one model that is installed in Ollama.", 400)
    levels = clean_levels(data.get("levels")) if test == "levels" else None
    if test == "levels" and not levels:
        raise LabError("Pick at least one difficulty level.", 400)
    with CONTROL_LOCK:
        if JOB.running():
            raise LabError("Something is already running. Wait for it to finish first.")
        JOB.queue = []
        requests = [{"model": m, "set": test, "levels": levels} for m in models]
        JOB.local_queue = requests[1:]
        return launch_local_run(requests[0])


def create_variant(data: dict) -> dict:
    base, name, system = str(data.get("base", "")), str(data.get("name", "")).strip().lower(), str(data.get("system", "")).strip()
    if base not in local_models_info()["models"]:
        raise LabError("Pick a base model that is installed in Ollama.", 400)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,40}", name):
        raise LabError("Use a short name with lowercase letters, digits, dots or dashes (e.g. qwen3-coder-rules).", 400)
    full = f"{name}:latest"
    if full in local_models_info()["models"]:
        raise LabError("A model with this name already exists.", 409)
    if not 10 <= len(system) <= 4000:
        raise LabError("Write the instructions for the new model (10–4,000 characters).", 400)
    try:
        local_models.create_variant(full, base, system)
    except local_models.ModelError as error:
        raise LabError(f"Ollama could not create the model: {error}", 502) from None
    with CONTROL_LOCK:
        variants = lab_variants()
        variants[full] = {"base": base, "system": system, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        LOCAL.mkdir(parents=True, exist_ok=True)
        VARIANTS.write_text(json.dumps(variants, indent=2) + "\n")
    return {"ok": True, "model": full}


def delete_variant(data: dict) -> dict:
    name = str(data.get("name", ""))
    variants = lab_variants()
    if name not in variants:
        raise LabError("Only models made in the lab can be deleted here.", 403)
    if JOB.running():
        raise LabError("Wait until the running test finishes.")
    try:
        local_models.delete_model(name)
    except local_models.ModelError as error:
        if "not found" not in str(error).lower():
            raise LabError(f"Ollama could not delete the model: {error}", 502) from None
    with CONTROL_LOCK:
        variants.pop(name, None)
        VARIANTS.write_text(json.dumps(variants, indent=2) + "\n")
    return {"ok": True}


def small_model_reference() -> dict:
    """The small model's best finished results, shown next to local models on the same test sets."""
    best: dict = {"course": None, "levels": []}
    for record in all_experiments():
        tests = eval_both(record)
        if not tests["after"]:
            continue
        summary = tests["after"]["summary"]
        row = {"name": record["name"], "solved": summary["tasks_solved"], "total": summary["tasks_total"],
               "levels": record.get("task_levels")}
        if kind_of(record) == "course":
            if not best["course"] or row["solved"] > best["course"]["solved"]:
                best["course"] = row
        else:
            row["by_level"] = {str(l["level"]): [l["after_solved"], l["after_total"]] for l in level_results(record)["levels"]}
            best["levels"].append(row)
    return best


def archive_local_run(data: dict) -> dict:
    run_id = str(data.get("id", ""))
    folder = LOCAL / run_id
    if not re.fullmatch(r"[a-z0-9-]+", run_id) or not folder.is_dir():
        raise LabError("Unknown local-model test.", 404)
    if JOB.running(f"local:{run_id}"):
        raise LabError("This test is running. Stop it first.")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.move(str(folder), str(ARCHIVE / f"local-{run_id}"))
    return {"ok": True}


def run_try(request: dict) -> dict:
    """Run lab/try_task.py in its own process with a time limit; model and user code never run in the server."""
    try:
        done = subprocess.run([sys.executable, "lab/try_task.py"], input=json.dumps(request), cwd=ROOT,
                              capture_output=True, text=True, timeout=TRY_TIMEOUTS[request["mode"]])
    except subprocess.TimeoutExpired:
        raise LabError("That took too long and was stopped.", 504) from None
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise LabError(f"Could not run the task: {(done.stderr or 'no output').strip().splitlines()[-1][:300]}", 500)
    answer = json.loads(lines[-1])
    if not answer.get("ok"):
        raise LabError(answer.get("error", "Failed."), 400)
    return answer


def try_snapshots() -> list[dict]:
    """Final snapshots of finished runs, newest first: what "Ask the small model" can choose from."""
    rows = []
    for record in reversed(all_experiments()):
        p, n = paths_for(record), plan(record)
        levels = record.get("task_levels") or [1]
        for label, path in ((f"{record['name']} · after RL", p.rl / f"checkpoint-{n['groups']:04d}"),
                            (f"{record['name']} · after pretraining", p.pretrain / f"checkpoint-{n['rl_start']:04d}")):
            if (path / "model.pt").exists():
                rows.append({"label": label, "checkpoint": str(path.relative_to(ROOT)), "levels": levels})
    return rows


def try_route(data: dict) -> dict:
    mode = data.get("mode")
    if mode not in ("show", "check", "ollama", "snapshot"):
        raise LabError("Unknown mode.", 400)
    family, index = data.get("family"), data.get("index")
    if family not in ladder.family_levels() or not isinstance(index, int) or not 0 <= index < 4:
        raise LabError("Pick a task.", 400)
    request = {"mode": mode, "family": family, "index": index}
    if mode == "check":
        code = str(data.get("code", ""))
        if not code.strip() or len(code) > 4000:
            raise LabError("Write a function first (up to 4,000 characters).", 400)
        request["code"] = code
    if mode == "ollama":
        request["model"] = str(data.get("model", ""))
    if mode == "snapshot":
        allowed = {row["checkpoint"] for row in try_snapshots()}
        if data.get("checkpoint") not in allowed:
            raise LabError("Pick a snapshot.", 400)
        request["checkpoint"] = data["checkpoint"]
    return run_try(request)


def save_custom_task(data: dict) -> dict:
    task = run_try({"mode": "validate", "description": str(data.get("description", ""))[:200],
                    "code": str(data.get("code", ""))[:4000], "tests": str(data.get("tests", ""))[:4000]})["task"]
    with CONTROL_LOCK:
        items = read_json_list(ladder.CUSTOM_FILE)
        items = [item for item in items if item["name"] != task["name"]] + [task]
        ladder.CUSTOM_FILE.write_text(json.dumps(items, indent=2) + "\n")
    return {"ok": True, "task": task}


def delete_custom_task(data: dict) -> dict:
    with CONTROL_LOCK:
        if JOB.running():
            raise LabError("Wait until the running stage finishes.")
        items = read_json_list(ladder.CUSTOM_FILE)
        kept = [item for item in items if item["name"] != data.get("name")]
        if len(kept) == len(items):
            raise LabError("Unknown task.", 404)
        ladder.CUSTOM_FILE.write_text(json.dumps(kept, indent=2) + "\n")
    return {"ok": True}


def read_json_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------- web server

def known_stage(data: dict) -> str:
    stage = data.get("stage")
    if stage not in STAGE_ORDER:
        raise LabError("Unknown stage.", 400)
    return stage


def compare_route(data: dict) -> dict:
    record = experiment(data.get("exp"))
    snapshots = snapshots_for(record)
    keys = [k for k in data.get("snapshots", []) if k in snapshots and (snapshots[k][1] / "model.pt").exists()]
    family, index = data.get("family"), data.get("index")
    if family not in FAMILIES or not isinstance(index, int) or not 0 <= index < 8 or not keys:
        raise LabError("Invalid task or model selection.", 400)
    try:
        return COMPARE.run(record, family, index, keys)
    except Exception as error:  # show the reason on the page instead of failing silently
        raise LabError(f"Comparison failed: {error}", 500) from error


def start_route(data: dict) -> dict:
    start_stage(experiment(data.get("exp")), known_stage(data))
    return {"ok": True}


def run_all_route(data: dict) -> dict:
    return {"ok": True, "stages": run_all(experiment(data.get("exp")))}


def stop_route(data: dict) -> dict:
    JOB.stop()
    return {"ok": True}


def archive_route(data: dict) -> dict:
    return {"ok": True, "archived": archive_stage(experiment(data.get("exp")), known_stage(data))}


def create_route(data: dict) -> dict:
    return {"ok": True, "experiment": create_experiment(data)}


def archive_experiment_route(data: dict) -> dict:
    archive_experiment(experiment(data.get("exp")))
    return {"ok": True}


GET_ROUTES = {
    "/api/state": lambda q: lab_state(experiment(q.get("exp"))),
    "/api/pretrain": lambda q: pretrain_curve(experiment(q.get("exp"))),
    "/api/rl": lambda q: rl_groups(experiment(q.get("exp"))),
    "/api/eval": lambda q: eval_both(experiment(q.get("exp"))),
    "/api/levels": lambda q: level_results(experiment(q.get("exp"))),
    "/api/overview": lambda q: overview("ladder" if q.get("kind") == "ladder" else "course"),
    "/api/ladder": lambda q: ladder_info(),
    "/api/local-models": lambda q: local_models_info(),
    "/api/local-runs": lambda q: {"runs": local_runs(), "reference": small_model_reference(),
                                  "queued": [r["model"] for r in JOB.local_queue]},
    "/api/job": lambda q: lab_job(),
    "/api/local-run": lambda q: read_json(LOCAL / re.sub(r"[^a-z0-9-]", "", q.get("id", "")) / "result.json") or {},
    "/api/try-options": lambda q: {"snapshots": try_snapshots()},
}
POST_ROUTES = {
    "/api/start": start_route,
    "/api/run-all": run_all_route,
    "/api/stop": stop_route,
    "/api/archive": archive_route,
    "/api/compare": compare_route,
    "/api/experiments": create_route,
    "/api/experiments/archive": archive_experiment_route,
    "/api/local-runs": lambda data: {"ok": True, "run": start_local_run(data)},
    "/api/local-runs/archive": archive_local_run,
    "/api/local-models/create": create_variant,
    "/api/local-models/delete": delete_variant,
    "/api/try": try_route,
    "/api/custom-tasks": save_custom_task,
    "/api/custom-tasks/delete": delete_custom_task,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the Terminal quiet
        pass

    def _write(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._write(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _from_this_page(self) -> bool:
        """Only this page may use the API: blocks other websites (and DNS-rebinding tricks) in the same browser."""
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin.removeprefix("http://") in ALLOWED_HOSTS

    def _body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise LabError("Expected JSON.", 415)
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            raise LabError("Invalid request.", 400) from None
        if not isinstance(data, dict):
            raise LabError("Invalid request.", 400)
        return data

    def do_GET(self) -> None:
        if not self._from_this_page():
            return self._json({"error": "Forbidden"}, 403)
        url = urlparse(self.path)
        if url.path in ("/", "/index.html", "/slides.html", "/lab", "/tasks", "/models"):  # one page, three tabs
            return self._write(200, "text/html; charset=utf-8", PAGE.read_bytes())
        route = GET_ROUTES.get(url.path)
        if route is None:
            return self._json({"error": "Not found"}, 404)
        try:
            self._json(route({key: values[0] for key, values in parse_qs(url.query).items()}))
        except LabError as error:
            self._json({"error": str(error)}, error.status)

    def do_POST(self) -> None:
        if not self._from_this_page():
            return self._json({"error": "Forbidden"}, 403)
        route = POST_ROUTES.get(urlparse(self.path).path)
        if route is None:
            return self._json({"error": "Not found"}, 404)
        try:
            self._json(route(self._body()))
        except LabError as error:
            self._json({"error": str(error)}, error.status)


def main() -> int:
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        print(f"\n  Port {PORT} is already in use. The lab may already be open in another Terminal window.")
        print("  Click that window and press Ctrl + C. Then run this command again.\n")
        return 1
    url = f"http://{HOST}:{PORT}"
    print(f"\n  Code Model Lab is running:  {url}   (Task Lab: {url}/tasks · Models: {url}/models)")
    print("  Keep this Terminal window open. To stop: Ctrl + C\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
