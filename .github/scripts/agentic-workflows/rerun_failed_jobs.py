# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Re-run ONLY the failed jobs of an analysed GitHub Actions workflow run.

Used by the `rerun-failed-jobs` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/rerun-failed-jobs.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT and calls the GitHub
Actions rerun-failed-jobs endpoint via PyGithub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from common import github_client, handle_numeric_id, read_agent_item, require_env, resolve_repository

if TYPE_CHECKING:
    from github.WorkflowRun import WorkflowRun


def trigger_rerun(run: "WorkflowRun", run_id: str, repository: str) -> None:
    """Re-run only the failed jobs of ``run``, guarding against restart loops."""
    # If the run has already been attempted more than once, do not re-run it
    # again (mirrors rerunner.py).
    attempt = run.run_attempt or 1
    if attempt > 1:
        print(f"Run {run_id} already has {attempt} attempts; not re-running to avoid loops.")
        return

    if not run.rerun_failed_jobs():
        raise SystemExit(f"GitHub API rejected the re-run request for {repository} run {run_id}.")

    print(f"Successfully requested re-run of failed jobs for {repository} run {run_id}.")


def main() -> None:
    item = read_agent_item("rerun_failed_jobs")
    run_id = handle_numeric_id(item.get("run_id") or "", "run_id")
    repository = resolve_repository(item.get("repository") or "")
    reason = item.get("reason") or ""

    print(f"Requested re-run of failed jobs for {repository} run {run_id} (reason: {reason})")

    token = require_env("GH_TOKEN", "GITHUB_TOKEN is not configured; cannot re-run failed jobs.")
    run = github_client(token).get_repo(repository).get_workflow_run(int(run_id))
    trigger_rerun(run, run_id, repository)


if __name__ == "__main__":
    main()
