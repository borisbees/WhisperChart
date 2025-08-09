# Collaboration Charter: Cursor + Codex

This repository is set up for two AI collaborators:

- Codex CLI agent ("Codex"): this terminal-based assistant operating with plans and patches.
- Cursor editor assistant ("Cursor"): in-editor partner for quick, focused edits.

Use this guide to coordinate smooth hand-offs and predictable workflows.

## Goals
- Keep changes safe, minimal, and reviewable.
- Bias toward small diffs with clear intent and easy rollback.
- Prefer local, testable improvements over broad refactors unless planned.

## Commands & Environment
- Run app: `make run` (starts `streamlit run app/app.py`)
- Install deps: `make install`
- Create venv: `make venv`
- Cleanup: `make clean`
- Secrets live in `app/.streamlit/secrets.toml` (see example in same folder). Do not hardcode secrets.

## Roles
- Codex
  - Multi-file refactors and architecture changes
  - Dependency and Makefile updates
  - Data/API schema adjustments and security-sensitive logic
  - Writing/curating repo-level docs and plans
- Cursor
  - Small, local refactors and bug fixes
  - Docstrings, typing, and logging improvements
  - Minor UI tweaks inside `app/app.py`
  - Inline test snippets or quick sanity checks

## Workflow
1. Propose: For non-trivial work, propose a short plan with the minimal file set to change.
2. Implement: Keep diffs tight. Follow `.cursorrules` and existing code style.
3. Verify: Prefer manual or scripted checks using Makefile targets.
4. Review: If scope grows, pause and hand off to the other agent as needed.

## Handoffs
- Defer to Codex when a change spans directories, alters dependencies, or affects security/auth/data schema.
- Defer to Cursor for quick fixes, small UI polish, logging, and documentation nits.
- Record notable handoffs here (optional):
  - [ ] Date — Area — From → To — Reason/notes

## Prompts (Cursor)
- General: "Follow .cursorrules and AGENTS.md. Propose a 2–4 step plan. Keep diffs minimal and focused. Use Makefile to run."
- UI tweak: "In `app/app.py`, add a small, optional UI control (no new deps). Include a quick verification note."
- Refactor nudge: "Suggest a local refactor to reduce complexity in function X, then apply it if risk is low."

## Prompts (Codex)
- Planning: "Outline plan and change list before edits. Apply patches with minimal scope."
- Repo docs: "Update AGENTS.md or README when workflows change."

## Style & Safety
- Keep functions small and cohesive; avoid hidden side effects.
- Do not introduce new frameworks/libraries without plan/approval.
- Prefer incremental improvements and reversible changes.

