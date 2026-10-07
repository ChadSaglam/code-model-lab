# AGENTS.md — definition of done

Conventions for anyone contributing to **Code Model Lab**. A change is done only when it
meets every point below. Where this file and a more specific rule disagree, the stricter wins.

## Stack

- **Python** 3.12 and 3.14 are the tested versions (3.10+ works); the local environment lives in `.venv`.
- Pure Python. No packaging or compile step — the lab is run directly from the source tree.
- Runtime dependencies: `torch`, `numpy`, `matplotlib`, pinned in `requirements.txt`.
- The server (`lab/serve_lab.py`) is Python's standard-library HTTP server, bound to `127.0.0.1` only.
- Everything runs locally; the lab talks only to a local Ollama instance and sends data to no outside service.

## Running the checks locally

```bash
make setup     # create .venv and install the pinned requirements (first time only)
make check     # run every check below; stops on the first failure
make dev       # start the lab at http://127.0.0.1:8765
```

`make check` runs the unit test suite:

```bash
.venv/bin/python -m pytest -q
```

A full pass is `37 passed`. Eleven of those tests bind a local HTTP port; they pass on a normal
machine but cannot run where binding a port is blocked — skip them there, do not delete them.

## Which checks a change must pass

- **Unit tests** — `make check` is green for the area you touched (and the whole suite before a merge).
- There is **no separate lint, type-check or build** tool configured. Keep imports clean, match the
  surrounding style, and keep the code importable. Do not add a new tool to "pass" a check.
- New logic ships with unit tests in `tests/`. Tests that need a running model use the stand-in Ollama
  already in `tests/test_model_lab.py`; do not require a real model to run the suite.

## Where tests live

All tests are in `tests/`:

- `test_lab.py` — the model: tokenizer, attention, layer rhythm, experts, reference solutions.
- `test_model_lab.py` — Task Lab levels, level-5 validation, answer extraction, local-model runs,
  model queues and instruction variants (stand-in Ollama), settings validation, server safety rules.
- `test_vision.py`, `test_full_25m_vision_digit_pilot.py` — the optional image path.

## Branches and releases

- Work on branches named `agent/<date>-<topic>`. The owner reviews, stages, squashes, releases and pushes them.
- Never commit to `main`, never push, never open pull requests, never run `git config`.
- Each merge is preceded by a demo branch; the owner runs the pre-merge check and deploys. Deploy is a human act.
- Internal communication is in English. Commit identity stays `Chad Saglam <saglam.chad@proton.me>`.

## Commit message style

- Conventional prefix: `feat(scope): …`, `fix(scope): …`, `test(scope): …`, `docs: …`, `chore: …`.
- Imperative subject ≤ 72 characters; the body says *what* and *why* in plain words.
- Reference a roadmap id (`B-xx`) when a change clearly serves one.

## What never changes without the owner

- **Dependencies** — `requirements.txt` and the versions it pins. A change that truly needs a new one stops and asks.
- **Data** — only synthetic test data belongs in the repo. No customer or real data in code, tests, fixtures or logs
  until the owner approves it in writing; real data stays on the owner's machines.
- **Legal and user-facing text** — licence, prices, published copy.
- **Release and automation config** — CI workflows and anything under `.github/`.
- Never add AI attribution or co-author lines anywhere visible in the repo, commits or history.
