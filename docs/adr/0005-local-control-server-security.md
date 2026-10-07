# ADR-005: Security model for a server that can start processes

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

The lab server can start training processes and move result folders. It runs on the user's computer while they browse other websites. A malicious site could try to send requests to `127.0.0.1` from the same browser (cross-site requests, DNS rebinding). The tests also execute model-written Python code.

## Decision

- **Listen only on `127.0.0.1`.** The server is not reachable from the network.
- **Accept requests only from its own page.** Every request must have `Host: 127.0.0.1:8765` (or `localhost:8765`); an `Origin` header, if present, must match. This blocks other websites and DNS-rebinding tricks.
- **JSON-only actions.** POST requests must be `application/json`, which browsers can't send cross-site without a preflight the server never approves.
- **No arbitrary commands.** The page can only choose among fixed stages; settings are validated numbers within fixed ranges or fixed choices.
- **Archive, never delete.** "Archive" moves results to `runs/_archive/`.
- **Model-written and user-written code** runs through the course's verifier, which rejects imports, loops, attribute access and any call outside a small allow-list before executing. Code from local models, snapshots or the "Try a task" box runs in a separate process (`lab/try_task.py`, `lab/ollama_eval.py`) with a time limit, never inside the server.

## Options Considered

### Option A: Local-only server with Host/Origin checks (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low – ~15 lines |
| Cost | None |
| Protection | Network access, cross-site requests, DNS rebinding |

**Pros:** no login needed; protects the realistic threats for a local tool.
**Cons:** does not protect against other software already running on the same computer.

### Option B: Token in the URL or a login
**Pros:** also blocks other local software.
**Cons:** friction for a single-user tool; tokens end up in browser history.

### Option C: No protection (localhost only)
**Cons:** any website could start or archive runs through the browser. Rejected.

## Trade-off Analysis

The realistic attacker is a web page in the same browser, not local malware (which could read the files directly anyway). Host and Origin checks plus JSON-only actions close that path without adding friction.

## Consequences

- Easier: safe to keep the lab open while browsing.
- Harder: opening the page from another device on the network is not possible (by design).
- Revisit: if the lab is ever exposed beyond localhost, add authentication and HTTPS first.

## Action Items

1. [x] Host/Origin check on every request, JSON-only POSTs, validated settings.
2. [x] Tests for forbidden origins, wrong hosts and bad input.
3. [x] Separate process and time limit for any code that is not part of the project.
