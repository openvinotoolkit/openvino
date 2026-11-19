# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mark a transient merge-queue failure as eligible for re-queueing.

Single custom safe-output job for the CI Doctor MQ workflow
(.github/workflows/shared/agentic-workflows/remediate-transient-failure.md). The
agent calls it whenever the failure is transient (Infrastructure / Flaky Test /
Network / External Service).

This job resolves the PR's live merge-queue status. If it is still queued, it
re-runs only the failed jobs. It never re-adds a PR: instead it leaves a
trusted, current-head marker for the deterministic dequeue workflow to consume.

The default ``GITHUB_TOKEN`` re-runs failed jobs and posts the marker comment.
"""

from __future__ import annotations

import os
import re

from common import (
    github_client,
    handle_numeric_id,
    merge_queue_status,
    pr_number_from_merge_queue_branch,
    read_agent_item,
    require_env,
    resolve_repository,
)

from github.PullRequest import PullRequest
from github.WorkflowRun import WorkflowRun


ELIGIBLE_MARKER = (
    "<!-- ci-doctor-mq-eligible pr={pr_number} head_sha={head_sha} "
    "source_run_id={source_run_id} analysis_run_id={analysis_run_id} -->"
)
SHA_RE = re.compile(r"[0-9a-f]{40}")


def _eligibility_comment(
    run: "WorkflowRun",
    pull: "PullRequest",
    reason: str,
    source_run_id: str,
    analysis_run_id: str,
    queue_status: str,
    rerun_requested: bool,
) -> str:
    """Build the visible explanation and machine-readable marker for the PR."""
    head_sha = pull.head.sha if pull.head else ""
    if not SHA_RE.fullmatch(head_sha or ""):
        raise SystemExit(f"PR #{pull.number} has no valid current head SHA; cannot mark it for re-queueing.")

    marker = ELIGIBLE_MARKER.format(
        pr_number=pull.number,
        head_sha=head_sha,
        source_run_id=source_run_id,
        analysis_run_id=analysis_run_id,
    )
    if rerun_requested:
        action = "CI Doctor re-ran the failed jobs because the PR is still in the merge queue."
    elif queue_status == "in_queue":
        action = (
            "The PR remains in the merge queue; CI Doctor did not request another retry because "
            "the run had already been retried."
        )
    else:
        action = (
            "The PR is no longer in the merge queue; the separate re-queue workflow will act only "
            "if GitHub confirms it was dequeued because of `CI_FAILURE`."
        )
    return (
        "### CI Doctor — Transient Merge Queue Failure\n\n"
        "CI Doctor classified this failure as transient and marked this PR as eligible for automatic "
        "re-queueing if it was removed from the merge queue because of `CI_FAILURE`. "
        f"{action}\n\n"
        f"- **Failed run**: {run.html_url}\n"
        f"- **Reason**: {reason or 'transient failure'}\n"
        f"\n{marker}\n"
    )


def resolve_pr_number(run: "WorkflowRun") -> str:
    """Best-effort PR number behind a merge-queue run, or '' when none applies."""
    number = pr_number_from_merge_queue_branch(run.head_branch or "")
    if number:
        return number
    try:
        pulls = run.pull_requests or []
    except Exception:
        pulls = []
    return str(pulls[0].number) if pulls else ""


def rerun_failed_jobs(run: "WorkflowRun", run_id: str, repository: str) -> bool:
    """Re-run only the failed jobs of ``run`` (loop-guarded)."""
    attempt = run.run_attempt or 1
    if attempt > 1:
        print(f"Run {run_id} already has {attempt} attempts; not re-running to avoid loops.")
        return False
    if not run.rerun_failed_jobs():
        raise SystemExit(f"GitHub API rejected the re-run request for {repository} run {run_id}.")
    print(f"Successfully requested re-run of failed jobs for {repository} run {run_id}.")
    _set_github_env("RERUN_FAILED_JOBS", "true")
    return True


def _set_github_env(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_ENV`` so later workflow steps can react."""
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def main() -> None:
    item = read_agent_item("remediate_transient_failure")
    run_id = handle_numeric_id(item.get("run_id") or "", "run_id")
    repository = resolve_repository(item.get("repository") or "")
    reason = item.get("reason") or ""

    print(f"Marking transient failure for {repository} run {run_id} (reason: {reason})")

    token = require_env("GH_TOKEN", "GITHUB_TOKEN is not configured; cannot remediate.")
    gh = github_client(token)
    repo = gh.get_repo(repository)
    run = repo.get_workflow_run(int(run_id))

    pr_number = resolve_pr_number(run)
    if not pr_number:
        print("No pull request associated with this run; cannot remediate.")
        return

    pull = repo.get_pull(int(pr_number))
    if pull.merged or pull.state != "open" or pull.draft:
        print(f"PR #{pr_number} is not an open, non-draft pull request; no marker will be posted.")
        return
    status = merge_queue_status(gh, repo, pull)
    print(f"Live merge-queue status for PR #{pr_number}: {status}")

    rerun_requested = False
    if status == "in_queue":
        print(f"PR #{pr_number} is still in the merge queue; re-running failed jobs.")
        rerun_requested = rerun_failed_jobs(run, run_id, repository)
    elif status == "not_in_queue":
        print(f"PR #{pr_number} is no longer in the merge queue; posting transient-eligibility marker only.")
    elif status == "merged":
        print(f"PR #{pr_number} is already merged; no eligibility marker is needed.")
        return
    else:
        print(f"PR #{pr_number} merge-queue status is '{status}'; cannot safely mark it. Skipping.")
        return

    analysis_run_id = handle_numeric_id(os.environ.get("GITHUB_RUN_ID", ""), "analysis_run_id")
    pull.create_issue_comment(
        _eligibility_comment(run, pull, reason, run_id, analysis_run_id, status, rerun_requested)
    )
    _set_github_env("TRANSIENT_ELIGIBILITY_MARKED", "true")
    print(f"Recorded transient-eligibility marker on PR #{pull.number}.")


if __name__ == "__main__":
    main()
