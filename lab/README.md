# Lab

The local web app: Model Lab (`/`) runs training experiments for the course model, Task Lab (`/tasks`) tests it on harder tasks, and Models (`/models`) tests and improves models installed in Ollama.

## Start

From the project folder:

```
.venv/bin/python lab/serve_lab.py
```

The page opens at http://127.0.0.1:8765. Keep the Terminal window open; stop with Ctrl + C.

## Experiments

An experiment is one full run of the course (pretraining → RL → two tests) with its own settings.

- **Course baseline** is the run you made from the README. Its files stay where they are.
- **+ New experiment** lets you change a few settings: seeds, learning rates, RL temperature, attempts per task. Everything else stays fixed so results are comparable.
- **Reuse pretraining** skips stage 1 and only changes RL. This is the fastest way to test RL ideas.
- **Run all remaining** runs every unfinished stage in order, one after the other.
- Each experiment is stored in `runs/experiments/<id>/` with an `experiment.json` that records its settings and the Python/PyTorch versions used.

## Task Lab (http://127.0.0.1:8765/tasks)

Harder tasks for the same model, to see where it stops keeping up.

- **Level 1** is the course's one-line tasks. **Level 2** adds two inputs or two operations, **level 3** if-statements, **level 4** small multi-step functions.
- Every task has hidden tests and stays inside the course verifier's safety rules (no loops, imports or attribute access).
- A run trains one model on the levels you pick. Context and answer length grow automatically to fit the hardest level.
- "Where the model breaks" shows the share of unseen tasks solved per level, before and after RL.
- The task definitions live in `lab/ladder.py`. It plugs them into the unchanged course scripts; the course files are not modified.

## Level 5, trying tasks and the Models tab

- **Level 5 · Your tasks:** add a task on the Task Lab page (description, reference function, tests). It is validated and saved to `lab/custom_tasks.json`.
- **Try a task yourself:** run your own code, a local model or a saved snapshot against one task and see every test case. Runs in `lab/try_task.py`, in its own process with a time limit.
- **Models (http://127.0.0.1:8765/models):** tick installed Ollama models and test them one after another; `lab/ollama_eval.py` runs each test set and records tokens read/written per answer. *Improve a model* creates a variant with its own instructions; variants are listed in `runs/local-models/variants.json` and are the only models the lab can delete. Shared helpers live in `lab/local_models.py`.

## Page sections

| Section | What you see |
|---|---|
| Difficulty levels (Task Lab) | What each level contains, with an example task |
| Stages | Start, stop and redo the selected run's stages, with live progress |
| Compare | All runs in one table, plus overlaid loss and RL success curves |
| Where the model breaks (Task Lab) | Unseen tasks solved per level, before and after RL |
| How this run improved | Loss curve and correct RL attempts per round |
| Before / after | The same unseen tasks, solved before and after RL |
| Development journey (Model Lab) | One task answered by every saved snapshot of the experiment |
| Try a task yourself (Task Lab) | Your code or a model's answer, case by case |
| Installed models, Leaderboard, Improve a model (Models) | Ollama models on the same tests, ranked, with tokens and speed; instruction variants |

## Safety

- "Archive" moves results to `runs/_archive/`. Nothing is deleted.
- Only one stage runs at a time.
- The server only listens on your own computer and only accepts requests from its own page.
