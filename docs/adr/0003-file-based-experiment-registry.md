# ADR-003: Store experiments as JSON files next to their results

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

An experiment is one full run (pretraining, RL, two tests) with its own settings. Comparing experiments is only meaningful if each one records exactly what it changed and on which software it ran. Results include large checkpoint files (~100 MB each) that stay local.

## Decision

Each experiment gets a folder `runs/experiments/<id>/` with an `experiment.json` (name, settings, levels, which pretraining it reuses, Python and PyTorch versions, a fixed color) next to its checkpoints, receipts and logs. The run made from the course README is shown as "Course baseline" at its original paths, without moving files. Removing an experiment moves its folder to `runs/_archive/`; nothing is deleted.

## Options Considered

### Option A: JSON file per experiment (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | None |
| Scalability | Dozens of experiments; fine for one person |
| Familiarity | Readable in any editor |

**Pros:** settings live next to the results they produced; copying or archiving a folder moves everything together; no database to corrupt or migrate.
**Cons:** listing experiments reads every file (fast at this size); no queries.

### Option B: SQLite database
**Pros:** queries, transactions.
**Cons:** settings and results can drift apart; a schema to maintain; harder to inspect by hand.

### Option C: MLflow or Weights & Biases
**Pros:** industry-standard experiment tracking, rich comparison UIs.
**Cons:** new dependency and server (MLflow) or a cloud account that uploads data (W&B); would need changes inside the course scripts to log metrics.

## Trade-off Analysis

The course scripts already write complete, self-describing receipts. A JSON record per experiment adds only what the receipts don't know (name, purpose, environment) and keeps everything together on disk. A tracking platform becomes worth it when many people share results, which is not the case here.

## Consequences

- Easier: understanding a result months later, backing up or sharing one experiment, reproducing it.
- Harder: cross-experiment queries beyond the comparison table.
- Revisit: move to MLflow if experiments are run on several machines or shared with others.

## Action Items

1. [x] Record Python and PyTorch versions in every experiment.
2. [x] Archive instead of delete; refuse to archive pretraining that another experiment reuses.
3. [ ] Add an "export experiment" button that zips the folder without checkpoints.
