# ADR-008: Release-gate thresholds for the B1/B2 packs

**Status:** Accepted
**Date:** 2026-10-25
**Deciders:** Project owner

## Context

B-12 added the mechanism: a pack run reports pass or fail against a threshold, and the
threshold is never hardcoded or defaulted. This record holds the values the owner chose.

The target user of the bookkeeping product does not know bookkeeping. They upload a PDF or a
photo and accept what they are shown. A person who cannot judge an account number will accept
a wrong one, so an average score alone says too little: what matters is how often the model is
wrong while it looks sure.

## Decision

A model or prompt change may be used in the product only when all of these hold:

1. **Floor:** B1 (bookkeeping) ≥ 90 %, B2 (invoices) ≥ 95 % — pass `--threshold 90` and
   `--threshold 95` to the respective pack runs.
2. **No regression:** the candidate scores at least as high as the model in use today, on both
   packs.
3. **Confident means right:** among answers the model marks as confident, ≥ 99 % are correct
   (measured by B-13, the false-confident rate). When the model is not sure, it does not guess:
   it asks the user, for example what an invoice was for, or points out a missing amount or date.

Real customer data, where the owner has consent, is used only on the owner's machine as an
additional holdout check. It is never stored in this repository and is not part of the gate.

## Consequences

- The model will propose fewer bookings automatically, and those it proposes are almost never
  wrong; the rest becomes a short question to the user.
- B-13 is required before rule 3 can be checked; until then rules 1 and 2 gate alone.
- Rule 2 needs the current model's scores on record (B-15).
