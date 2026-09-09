# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remediate a transient merge-queue pipeline failure.

Single custom safe-output job for the CI Doctor MQ workflow
(.github/workflows/shared/agentic-workflows/remediate-transient-failure.md). The
agent calls it whenever the failure is transient (Infrastructure / Flaky Test /
Network / External Service).

This job resolves that status live (``common.merge_queue_status``) and picks the
remedy itself:
  - PR still ``in_queue``   -> re-run only the failed jobs (a re-run keeps it moving).
  - PR ``not_in_queue``     -> re-add the PR to the merge queue (it was dropped).
  - PR ``merged``/``unknown`` or no PR that must be re-added -> no-op.

In both the re-run and re-add cases it posts an outcome comment on the PR so the
author sees what CI Doctor did and why.

Re-running uses the default ``GITHUB_TOKEN`` (needs ``actions: write`` and
``pull-requests: write`` for the comment); re-adding uses ``MERGE_QUEUE_TOKEN``
because the default token cannot re-trigger ``merge_group`` check runs.
"""

from __future__ import annotations

import os
# Merge-queue remediation: gh called with a fixed argv; PR/repo ids validated in common.py.
import subprocess  # nosec B404

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


READD_MARKER = "<!-- ci-doctor-mq-readd -->"


def _rerun_comment(run: "WorkflowRun", reason: str) -> str:
    """Comment body announcing that the PR's failed jobs were re-run."""
    return (
        "### CI Doctor — Merge Queue remediation\n\n"
        "This PR's merge-queue pipeline failed transiently, so CI Doctor **re-ran its failed jobs** "
        "(the PR is still in the merge queue).\n\n"
        f"- **Failed run**: {run.html_url}\n"
        f"- **Reason**: {reason or 'transient failure'}\n"
    )


def _readd_comment(run: "WorkflowRun", reason: str) -> str:
    """Comment body announcing that the PR was re-added to the merge queue."""
    return (
        f"{READD_MARKER}\n\n"
        "### CI Doctor — Merge Queue remediation\n\n"
        "This PR was dropped from the merge queue by a transient pipeline failure, so CI Doctor "
        "**re-added it to the merge queue**.\n\n"
        f"- **Failed run**: {run.html_url}\n"
        f"- **Reason**: {reason or 'transient failure'}\n"
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


def rerun_failed_jobs(run: "WorkflowRun", run_id: str, repository: str, pull: "PullRequest", reason: str) -> None:
    """Re-run only the failed jobs of ``run`` (loop-guarded) and comment on the PR."""
    attempt = run.run_attempt or 1
    if attempt > 1:
        print(f"Run {run_id} already has {attempt} attempts; not re-running to avoid loops.")
        return
    if not run.rerun_failed_jobs():
        raise SystemExit(f"GitHub API rejected the re-run request for {repository} run {run_id}.")
    print(f"Successfully requested re-run of failed jobs for {repository} run {run_id}.")
    _set_github_env("RERUN_FAILED_JOBS", "true")
    pull.create_issue_comment(_rerun_comment(run, reason))
    print(f"Recorded re-run comment on PR #{pull.number}.")


def _already_readded(pull: "PullRequest") -> bool:
    """Return True if a previous CI Doctor re-add marker comment is present."""
    return any(READD_MARKER in (comment.body or "") for comment in pull.get_issue_comments())


def _set_github_env(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_ENV`` so later workflow steps can react."""
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _readd_skip_reason(pull: "PullRequest", pr_number: str) -> str | None:
    """Return a reason to skip re-adding, or None to proceed."""
    if pull.merged:
        return f"PR #{pr_number} is already merged; nothing to re-add."
    if pull.state != "open":
        return f"PR #{pr_number} is not open (state={pull.state}); not re-adding."
    if pull.draft:
        return f"PR #{pr_number} is a draft; not re-adding."
    return None


def readd_to_queue(pr_number: str, repository: str, reason: str, merge_queue_token: str, run: "WorkflowRun") -> None:
    """Re-add a dropped PR to the merge queue via ``gh pr merge`` (idempotent).

    Uses ``MERGE_QUEUE_TOKEN`` for the enqueue and the outcome comment because the
    default ``GITHUB_TOKEN`` cannot re-trigger ``merge_group`` check runs.
    """
    mq_pull = github_client(merge_queue_token).get_repo(repository).get_pull(int(pr_number))

    if _already_readded(mq_pull):
        print(f"PR #{pr_number} already re-added by CI Doctor (marker comment found); skipping.")
        return

    reason_to_skip = _readd_skip_reason(mq_pull, pr_number)
    if reason_to_skip:
        print(reason_to_skip)
        return

    # ``--squash`` keeps the CLI non-interactive; on a merge-queue branch the
    # server ignores it in favour of the queue's configured method.
    merge_command = ["gh", "pr", "merge", pr_number, "--repo", repository, "--squash"]
    if mq_pull.head.sha:
        merge_command += ["--match-head-commit", mq_pull.head.sha]
    result = subprocess.run(
        merge_command, capture_output=True, text=True, env={**os.environ, "GH_TOKEN": merge_queue_token}
    )
    if result.returncode != 0:
        raise SystemExit(f"Merge-queue enqueue failed: {result.stderr.strip()}")
    print(f"Successfully requested re-add of PR #{pr_number} to the merge queue.")
    _set_github_env("READDED_TO_MQ", "true")

    mq_pull.create_issue_comment(_readd_comment(run, reason))
    print(f"Recorded re-add comment on PR #{pr_number}.")


def main() -> None:
    item = read_agent_item("remediate_transient_failure")
    run_id = handle_numeric_id(item.get("run_id") or "", "run_id")
    repository = resolve_repository(item.get("repository") or "")
    reason = item.get("reason") or ""

    print(f"Remediating transient failure for {repository} run {run_id} (reason: {reason})")

    token = require_env("GH_TOKEN", "GITHUB_TOKEN is not configured; cannot remediate.")
    gh = github_client(token)
    repo = gh.get_repo(repository)
    run = repo.get_workflow_run(int(run_id))

    pr_number = resolve_pr_number(run)
    if not pr_number:
        print("No pull request associated with this run; cannot remediate.")
        return

    pull = repo.get_pull(int(pr_number))
    status = merge_queue_status(gh, repo, pull)
    print(f"Live merge-queue status for PR #{pr_number}: {status}")

    if status == "in_queue":
        print(f"PR #{pr_number} is still in the merge queue; re-running failed jobs.")
        rerun_failed_jobs(run, run_id, repository, pull, reason)
    elif status == "not_in_queue":
        print(f"PR #{pr_number} was dropped from the merge queue; re-adding it.")
        merge_queue_token = require_env(
            "MERGE_QUEUE_TOKEN", "MERGE_QUEUE_TOKEN secret is not configured; cannot re-add to merge queue."
        )
        readd_to_queue(pr_number, repository, reason, merge_queue_token, run)
    elif status == "merged":
        print(f"PR #{pr_number} is already merged; no remediation needed.")
    else:
        print(f"PR #{pr_number} merge-queue status is '{status}'; cannot safely remediate. Skipping.")


if __name__ == "__main__":
    main()
