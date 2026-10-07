# ADR-004: Add task levels at runtime instead of editing the course

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

Task Lab adds harder task types (levels 2–4) to see where the 25.7M-parameter model stops keeping up. The course defines its eight task types in `glm53_flash/tasks.py`, and its training and test scripts read them from there. Multi-line answers also need a different rule for when generation stops: the course stops as soon as the code parses, which would cut an `if` function after its first branch.

## Decision

`lab/ladder.py` defines the new task types (as plain data, with hidden tests) and acts as a runner: it replaces the task list and the stop rule inside the already-imported course modules, then runs the unchanged course script. Level 1 is exactly the course's task list, so level-1 test tasks are identical to the course's. Context length and answer length are computed from the chosen levels.

## Options Considered

### Option A: Runtime injection through a runner (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low – one file |
| Cost | Relies on module attribute names in the course code |
| Scalability | Adding a task type = adding one row |
| Familiarity | Plain Python |

**Pros:** course files untouched; Model Lab results unaffected; levels can be combined freely.
**Cons:** depends on internal names (`FAMILIES`, `completion_is_parseable`); a course refactor could break it silently.

### Option B: Edit `glm53_flash/tasks.py` directly
**Pros:** simplest to read.
**Cons:** changes the course's own results and tests; every course experiment would need re-running.

### Option C: Copy the scripts into `lab/` and modify them
**Pros:** full control.
**Cons:** two copies of the training code that drift apart.

## Trade-off Analysis

Keeping the course reproducible matters more than avoiding a small dependency on internal names. That risk is contained by a check that every reference solution passes its own tests and that level-1 tasks match the course exactly.

## Consequences

- Easier: adding task types; running any mix of levels.
- Harder: upgrading the course code without checking the injected names.
- **Known limitation:** every task of a type has the same reference solution; only the function name and wording change. The tests therefore measure whether the model maps a description to the right pattern, not whether it can reason about new problems. High scores here do not mean general coding ability.

## Action Items

1. [x] Test that every reference solution passes and level 1 equals the course.
2. [ ] Add task types with varying constants (e.g. "add 7", "clamp to 0–25") so answers differ per task.
3. [ ] Hold out whole task types from training and test only on those, to measure real generalisation.
