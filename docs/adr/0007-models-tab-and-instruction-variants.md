# ADR-007: A separate Models tab; improve local models with instructions before training them

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

Testing Ollama models started as one section at the bottom of Model Lab and Task Lab. In practice it became its own activity: testing several models in a row, ranking them, reading their failed answers and trying to make them better. A section under the training pages hid that work and tested only one model at a time.

"Improving" a local model can mean three very different things: changing its instructions (minutes, no training), fine-tuning it on examples (hours, needs a GPU-friendly training stack and a dataset), or training a model from scratch (what Model Lab already does for the small model).

## Decision

- **A third tab, Models (`/models`)**, on the same page and server. It lists installed models with family, size and quantization, tests any selection one model after another (a model-test queue next to the stage queue), and ranks finished tests in a leaderboard with the small model as a reference row.
- **Improve with instructions first.** *Improve a model* creates a new Ollama model from an installed one with its own system prompt (`POST /api/create` with `from` and `system`; older Ollama versions get an equivalent Modelfile). The lab's test rules moved from the request's system message into the prompt, so a variant's own instructions are not overwritten during tests.
- **Only lab-made models can be deleted.** They are recorded in `runs/local-models/variants.json`; the delete endpoint refuses every other name, so a model the user installed is never removed by the lab.
- **Fine-tuning is a later, separate step** (planned), measured on the same tasks and leaderboard.

## Options Considered

### Option A: Separate tab + instruction variants (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low – two endpoints, one queue, one page mode |
| Cost | Free; a variant reuses the base model's weights (a few KB) |
| Signal | Shows whether failures come from how the model is told to answer |

**Pros:** fast loop (create, test, compare in minutes); results are directly comparable with the original.
**Cons:** cannot teach the model anything it does not already know.

### Option B: Fine-tune immediately (LoRA)
**Pros:** real changes to what the model knows.
**Cons:** needs a dataset, extra dependencies and long runs; without a baseline of instruction-only results it is unclear what fine-tuning adds.

### Option C: Keep a section on each lab page
**Cons:** one model at a time, duplicated on two pages, and hidden under the training views.

## Trade-off Analysis

Many failures of large local models on these tasks are rule violations (loops, method calls), not missing knowledge. Instructions fix that class cheaply and establish the baseline a later fine-tune must beat.

## Consequences

- Easier: testing every installed model with one click; seeing whether a prompt change helps.
- Harder: a variant's name must not collide with an installed model; Ollama must be running to create or delete.
- Revisit: when fine-tuning is added, record which method made each model.

## Action Items

1. [x] Models tab with model details, multi-model queue, leaderboard and failed answers.
2. [x] Create and delete instruction variants; refuse to delete models not made in the lab.
3. [x] Tests with a stand-in Ollama for variants and the model queue.
4. [ ] Fine-tune a local model on your own code and compare it on the same leaderboard.
