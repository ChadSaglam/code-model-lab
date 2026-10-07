# ADR-001: A local web page served by Python's standard library

**Status:** Accepted
**Date:** 2026-10-07
**Deciders:** Project owner

## Context

Training runs take from a few minutes to about an hour on a laptop. Starting them from Terminal meant typing long commands, and the results were spread across JSON files. The goal was one place to start runs, watch progress and compare results, usable without the command line.

Constraints: it must run offline on a Mac, add no new dependencies to the course's `requirements.txt`, and work with any Python version the course supports.

## Decision

A single-page web app (`lab/index.html`, plain HTML/CSS/JavaScript) served by a small server built on `http.server.ThreadingHTTPServer` (`lab/serve_lab.py`). The page polls a JSON API every two seconds. Charts are hand-drawn SVG.

## Options Considered

### Option A: Standard-library server + plain HTML (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low – two files, no build step |
| Cost | None – no extra packages |
| Scalability | One user on one machine, which is all that is needed |
| Familiarity | Plain Python and JavaScript |

**Pros:** nothing to install, starts instantly, works offline, easy to read end to end.
**Cons:** routing, JSON handling and charts are written by hand; no live push (polling instead).

### Option B: FastAPI or Flask + a frontend framework
| Dimension | Assessment |
|---|---|
| Complexity | Medium – web framework, possibly a JS build |
| Cost | Extra dependencies to pin and update |
| Scalability | More than needed |
| Familiarity | High for web developers |

**Pros:** cleaner routing, validation, WebSockets.
**Cons:** new dependencies for a teaching repository, more moving parts than the problem needs.

### Option C: Streamlit or Gradio
**Pros:** very fast to build dashboards.
**Cons:** heavy dependencies, its own rerun model fits long background jobs poorly, less control over layout.

### Option D: Jupyter notebook
**Pros:** familiar to ML users.
**Cons:** still code-first, background jobs and live progress are awkward, not friendly for non-programmers.

## Trade-off Analysis

The lab is a single-user local tool. Its hardest problems are running long jobs safely and showing their progress, not serving many users. A framework would solve problems the lab does not have and add dependencies to a repository whose value is being small and readable. Polling every two seconds is simpler than WebSockets and fast enough for runs that last minutes.

## Consequences

- Easier: install (nothing new), reading the whole lab in one sitting, running on any Python 3.10+.
- Harder: adding many new pages; each view is hand-written HTML and SVG.
- Revisit: if the page grows beyond two modes, split the JavaScript into modules or adopt a small framework.

## Action Items

1. [x] Serve one page for both modes (`/` and `/tasks`).
2. [ ] Split `index.html` into separate CSS/JS files if it grows past ~1,500 lines.
