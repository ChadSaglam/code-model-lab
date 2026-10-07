# ADR-002: Run the unchanged course scripts as subprocesses

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

The course provides working, tested scripts for pretraining, reinforcement learning and evaluation. The lab needs to run them with different settings, show live progress, and stop them on request. The course results must stay reproducible with the course's own commands.

## Decision

The lab starts each stage as a separate process (`subprocess.Popen`) with the same command line the course README uses, plus the experiment's settings. It reads the script's printed JSON progress lines while it runs, and the receipt file when it finishes. Only one stage runs at a time; a queue starts the next stage when the previous one succeeds.

## Options Considered

### Option A: Subprocesses of the unchanged scripts (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | Process start-up of ~1–2 s per stage |
| Scalability | One job at a time (one GPU) |
| Familiarity | Same commands as the course README |

**Pros:** course code untouched; a crash in training cannot take down the page; stopping a run is a clean process stop; results are identical to running the command by hand.
**Cons:** progress only arrives as often as the script prints it (every 10 steps in pretraining).

### Option B: Import the training code and call it in a thread
**Pros:** finer-grained progress, shared model objects.
**Cons:** requires refactoring the course scripts into a library; a training error or out-of-memory kills the server; stopping a thread cleanly is hard in Python.

### Option C: A job queue (Celery, RQ) with workers
**Pros:** robust queuing, retries.
**Cons:** needs a broker such as Redis; far too much for one laptop and one job at a time.

## Trade-off Analysis

Process isolation is the deciding factor: training can fail in many ways (out of memory, bad settings), and the page should survive all of them and show the error. Keeping the scripts unchanged also means anyone can verify a lab result by running the printed command in Terminal.

## Consequences

- Easier: upgrading the course code, debugging a run (the exact command and its full log are kept in `_logs/`).
- Harder: anything that needs data the scripts don't print or save.
- Revisit: if live per-step metrics become important, have the scripts write a progress file instead of parsing printed lines.

## Action Items

1. [x] One job at a time, guarded by a lock so two clicks can't start two runs.
2. [x] "Run all remaining" queue that stops on the first failure.
