---
name: feedback-use-uv-run
description: Always use uv run for executing Python scripts and tests in this project
metadata:
  type: feedback
---

Always use `uv run` to execute Python commands in this project (e.g., `uv run pytest`, `uv run python`).

**Why:** The project uses uv for dependency management. Running pytest or python directly hits the wrong venv or a system Python that lacks the project's packages.

**How to apply:** Any time a test, script, or Python command needs to run in this repo, prefix it with `uv run`.
