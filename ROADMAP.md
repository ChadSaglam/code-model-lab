# Roadmap

Items are sized for one focused run each. Thresholds, release tags, repo settings and the use of
real data are the owner's calls and are not listed here. "Gate:" names what must be true first.

## Now
- [ ] B-01 — Lab copyright in LICENSE: the MIT licence keeps Vuk Rosić's line and adds "Copyright (c) 2026 Chad Saglam" for the lab additions.
- [ ] B-02 — Test workflow and status badge: a release workflow runs `pytest -q` on Python 3.12 with CPU torch, and the README shows its status badge; done when the run is green and the badge renders.

## Next
- [ ] B-05 — Bookkeeping B1 pack: 60–100 synthetic Swiss bank-transaction lines in German, each mapped to the correct KMU chart-of-accounts number; done when the pack loads, validates and reads like real statements.
- [ ] B-06 — Invoice B2 pack: 30–50 synthetic invoice texts carrying date, total, VAT rate and VAT amount as JSON fields; done when the pack loads and validates against the JSON-field check.
- [ ] B-07 — Per-pack leaderboard: per-pack score, accuracy per field, tokens and speed in the Models tab; done when B1 and B2 results rank there, with tests using the stand-in Ollama.

## Later
- [ ] B-08 — Buchhaltung discovery (Gate: B-07): read the Buchhaltung repo and write up its current stack and where an account suggestion fits into the booking flow; done when the stack and the insertion point are documented.
- [ ] B-09 — Account-suggestion service (Gate: B-08, legal check): a service behind a feature flag (off by default) that calls the local Ollama HTTP API with a timeout and a no-suggestion fallback; done when the flag toggles it and the fallback is tested.
- [ ] B-10 — Suggestion UI with confirm step (Gate: B-09): show the suggestion with a confidence label and Accept / Change / Reject; nothing is booked without a click; done when each action works and only a click books.
- [ ] B-11 — Suggestion audit log (Gate: B-09): record timestamp, user, input, suggested account, model, prompt version and final decision; done when every suggestion decision is logged and can be read back.
- [ ] B-12 — Release gate on B1/B2 (Gate: B-07): make the B1/B2 packs the shared pass/fail contract so a model or prompt change requires a lab run that meets the owner's threshold; done when a run reports pass or fail against a configured threshold.
- [ ] B-13 — False-confident rate (Gate: B-12): measure wrong answers shown with high confidence separately from accuracy; done when the leaderboard reports a false-confident rate per pack.
- [ ] B-14 — Varying constants and held-out task types: as noted in the README limitations and ADR-004, add tasks whose constants vary and task types held out from training; done when they run in Task Lab and the leaderboard reflects them.

## Done
- [x] B-03 — Text-answer task kind: `lab/text_tasks.py` adds a non-code `TextTask` and three pure checks — `exact_match` (equal after whitespace is collapsed), `json_fields_match` (answer parses as a JSON object and the named fields equal the expected ones) and `numeric_match` (answer parses as a finite number within a tolerance); a malformed JSON or non-numeric answer scores wrong instead of raising. Text tasks never touch the sandbox path (no `validate_source`, no `exec`): `local_models.answer_text_task`/`check_text`/`build_text_prompt` ask a local model and score its plain text by rule. Verified by unit tests in `tests/test_model_lab.py` covering all three checks (including malformed JSON, non-numeric and infinite answers, an answer that is Python code compared as text, and an unknown check kind rejected) plus a stand-in-Ollama wiring test; `make check` is the owner's gate.
- [x] B-04 — Task-pack file format: `lab/task_packs.py` defines a JSON pack (`id`, `name`, and a list of `tasks`, each with `id`, `input`, `expected` and a `check` that is one of the three B-03 kinds — `exact`/`json`/`numeric`, the last carrying a `tolerance`). `load_pack`/`validate_pack` check a pack as it is read, mirroring the level-5 validation style (`lab/try_task.py` validate, `lab/ladder.py` load_custom): a malformed pack raises a `ValueError` naming the offending task id and field (missing field, unknown check, duplicate id, non-numeric expected, misplaced tolerance, empty/absent fields, invalid JSON), and a valid pack becomes a `TaskPack` of B-03 `TextTask` objects ready for `text_tasks.score`. A small synthetic example pack lives in `tests/fixtures/example_pack.json` (no real data). Verified by unit tests in `tests/test_model_lab.py` covering a valid pack load, the on-disk fixture, and each rejection path; `make check` is the owner's gate.
