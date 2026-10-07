# ADR-006: Test local models and your own tasks with the same hidden tests

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

A score for the 25.7M-parameter model means little on its own. Two comparisons make it meaningful: how a much larger model does on exactly the same tasks, and how a person does. Larger models are already installed locally through Ollama (for example 8–14B parameter models). People also want to add tasks of their own ("level 5") and try any task by hand.

## Decision

- **Local models** are called through Ollama's HTTP API on `127.0.0.1:11434` (or `OLLAMA_HOST`). `lab/ollama_eval.py` runs a whole test set as a lab job; `lab/try_task.py` answers one task. Answers are deterministic (temperature 0, thinking switched off where supported). The function is extracted from the reply (markdown fences and explanations removed) and checked by the same verifier and tests as the small model. Ollama reports the tokens each answer read and wrote; the lab stores and shows them.
- **Your own tasks** (level 5) are written on the page as a reference function plus tests (`input -> expected`). Before saving, the reference must pass its own tests under the verifier's rules. Tasks are stored in `lab/custom_tasks.json` and become a normal task type for runs, tests and trying.
- **Trying a task** runs your code, a local model's answer or a saved snapshot's answer and shows every test case: input, expected value and what came back.

## Options Considered

### Option A: Ollama HTTP API with the course verifier (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low – standard library `urllib`, no new packages |
| Cost | Free; runs locally |
| Fairness | Same prompts, same tests, same safety rules as the small model |

**Pros:** works with any model the user installs; token counts come from Ollama itself.
**Cons:** chat models sometimes add explanations or use loops, which the verifier rejects. The prompt states the rules, and the result shows the reason.

### Option B: Load larger models directly with PyTorch / transformers
**Pros:** no separate app.
**Cons:** large downloads, new dependencies, memory management in the lab itself.

### Option C: Cloud APIs
**Pros:** strongest models.
**Cons:** costs money per token, sends tasks off the computer, needs keys. Out of scope for a local, free lab.

## Trade-off Analysis

The verifier's rules (no loops, imports or methods) were designed for safety, not for large models; they make some correct-looking answers fail. Keeping one rule set for everyone is what makes the numbers comparable, so the rules are stated in the prompt and shown in the results instead of being relaxed for large models.

## Consequences

- Easier: putting the small model's score next to an 8–14B model's on identical tasks; growing the task set without editing code.
- Harder: very slow models make long test sets slow (one answer at a time).
- Revisit: run several answers per task (pass@k) once single answers are well understood.

## Action Items

1. [x] Show tokens read and written per answer, per test set and per training stage.
2. [x] Validate level-5 tasks before saving; run all model and user code in a separate process with a time limit.
3. [ ] Optional: let level-5 tasks vary constants per test case, so answers cannot be memorised.
