# Architecture

This project has two layers:

1. **The course code** (`glm53_flash/`, `scripts/`, `experiments/`, `tests/`): a 25.7M-parameter GLM-style language model, its training scripts and its executable verifier. It comes from the open-source course and is kept unchanged.
2. **The lab** (`lab/`): a local web app that runs those scripts, tracks experiments and compares results. Everything added in this project lives here.

The lab never edits the course code. It only starts the course scripts with chosen arguments and reads the files they write. That keeps the course reproducible and makes the lab easy to remove or upgrade.

## Components

```text
Browser (http://127.0.0.1:8765)            lab/index.html
   │  JSON over HTTP, polled every 2 s
   ▼
lab/serve_lab.py  ── standard-library HTTP server, bound to 127.0.0.1
   │  ├─ experiment registry   runs/experiments/<id>/experiment.json
   │  ├─ job runner            one subprocess at a time + a stage queue + a model-test queue
   │  ├─ result readers        training receipts and test reports (JSON)
   │  └─ snapshot comparer     loads saved checkpoints on the CPU
   │
   ├──► python scripts/train_pretrain.py ...            (Model Lab)
   ├──► python lab/ladder.py --levels 1,2 scripts/...   (Task Lab)
   │        └─ swaps in the chosen task levels, then runs the same script
   ├──► python lab/ollama_eval.py --model qwen3:8b ...  (Models page, via Ollama)
   └──► python lab/try_task.py                          (one task: your code, a model or a snapshot)
   ▼
runs/  ── checkpoints, receipts, test reports, logs (ignored by git)
```

## Data flow of one run

| Stage | Command | Output read by the lab |
|---|---|---|
| 1 · Pretraining | `scripts/train_pretrain.py` | `pretrain/training-receipt.json` (loss per step, time) |
| 2 · Reinforcement learning | `scripts/train_rl.py` | `rl/training-receipt.json` (rewards per round, time) |
| 3 · Test before RL | `scripts/evaluate.py` | `eval-before.json` (answer and test result per task) |
| 4 · Test after RL | `scripts/evaluate.py` | `eval-after.json` |

While a stage runs, the lab reads the script's printed progress lines. When it finishes, the lab reads the receipt it wrote.

## Three tabs, one page

| | Model Lab (`/`) | Task Lab (`/tasks`) | Models (`/models`) |
|---|---|---|---|
| Purpose | Compare training settings on the course tasks | See how far the small model gets on harder tasks | Test, rank and improve models installed in Ollama |
| Tasks | 8 course task types | Up to 23 task types in 4 levels, plus your own (level 5) | Either of the two sets, same hidden tests |
| Runs | Course baseline + experiments | Task Lab runs | One test per model, several in a row |
| Extra views | Development journey (snapshots side by side) | Where the model breaks, Try a task | Leaderboard, failed answers, Improve a model |

The Models page stores its tests in `runs/local-models/<id>/` and the models it made in `runs/local-models/variants.json`; only models listed there can be deleted from the page.

## Decisions

| ADR | Decision |
|---|---|
| [ADR-001](adr/0001-local-stdlib-web-server.md) | A local web page served by Python's standard library |
| [ADR-002](adr/0002-run-course-scripts-as-subprocesses.md) | Run the unchanged course scripts as subprocesses |
| [ADR-003](adr/0003-file-based-experiment-registry.md) | Store experiments as JSON files next to their results |
| [ADR-004](adr/0004-task-levels-injected-at-runtime.md) | Add task levels at runtime instead of editing the course |
| [ADR-005](adr/0005-local-control-server-security.md) | Security model for a server that can start processes |
| [ADR-006](adr/0006-local-models-and-own-tasks.md) | Test local models and your own tasks with the same hidden tests |
| [ADR-007](adr/0007-models-tab-and-instruction-variants.md) | A separate Models tab; improve local models with instructions before training them |
