"""Tests for ollama_eval.summarize: the run-summary the leaderboard reads.

summarize folds a run's per-task episodes into the totals the Models page shows — how many
tasks were solved, the tokens read and written, the wall-clock seconds, and a throughput in
tokens per second. It is pure arithmetic over the episode list, so its boundaries matter:
an empty run, a model that answered so fast its measured time rounds to zero (the division
guard), and the fact that throughput is model time, not the run's wall clock. The rest of
ollama_eval drives a real Ollama model and is out of scope here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))

from ollama_eval import summarize  # noqa: E402  (import after sys.path is set)


def episode(*, passed=True, read=0, written=0, seconds=0.0):
    """One task's result, with only the fields summarize reads."""
    return {"passed": passed, "read": read, "written": written, "seconds": seconds}


# ---------------------------------------------------------------- shape

def test_summary_has_exactly_the_keys_the_page_reads():
    summary = summarize([episode()], 1.0)
    assert set(summary) == {"tasks_solved", "tasks_total", "read", "written",
                            "seconds", "tokens_per_second"}


# ---------------------------------------------------------------- counts

def test_only_passed_episodes_count_as_solved():
    episodes = [episode(passed=True), episode(passed=False), episode(passed=True)]
    summary = summarize(episodes, 1.0)
    assert summary["tasks_solved"] == 2
    assert summary["tasks_total"] == 3


def test_read_and_written_tokens_are_summed():
    episodes = [episode(read=10, written=3), episode(read=5, written=7)]
    summary = summarize(episodes, 1.0)
    assert summary["read"] == 15
    assert summary["written"] == 10


# ---------------------------------------------------------------- seconds

def test_wall_clock_seconds_is_rounded_to_one_decimal():
    assert summarize([episode()], 3.14159)["seconds"] == 3.1


def test_throughput_divides_written_tokens_by_model_seconds():
    # 10 tokens written over 2.0 s of model time is 5.0 tok/s — the wall clock (99 s) is not used.
    episodes = [episode(written=4, seconds=0.5), episode(written=3, seconds=0.5),
                episode(written=3, seconds=1.0)]
    assert summarize(episodes, 99.0)["tokens_per_second"] == 5.0


def test_throughput_is_rounded_to_one_decimal():
    # 10 tokens over 3.0 s of model time is 3.333…, reported as 3.3.
    episodes = [episode(written=4, seconds=1.0), episode(written=3, seconds=1.0),
                episode(written=3, seconds=1.0)]
    assert summarize(episodes, 5.0)["tokens_per_second"] == 3.3


# ---------------------------------------------------------------- boundaries

def test_empty_run_summarises_to_zeros_without_error():
    summary = summarize([], 0.0)
    assert summary == {"tasks_solved": 0, "tasks_total": 0, "read": 0, "written": 0,
                       "seconds": 0.0, "tokens_per_second": 0.0}


def test_zero_model_seconds_does_not_divide_by_zero():
    # A model that answered faster than the timer's resolution has zero measured seconds;
    # the guard keeps throughput a finite number instead of raising ZeroDivisionError.
    summary = summarize([episode(written=5, seconds=0.0)], 0.0)
    assert math.isfinite(summary["tokens_per_second"])
    assert summary["tokens_per_second"] > 0
