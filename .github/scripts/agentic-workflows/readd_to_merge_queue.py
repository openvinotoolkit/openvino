# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Re-add a pull request dropped from the merge queue back into the queue.

Used by the `readd-to-merge-queue` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/readd-to-merge-queue.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT, and re-adds the PR via
`gh pr merge` (which enqueues on a merge-queue branch). It is idempotent: it
skips merged/closed/draft PRs and PRs already carrying a re-add marker comment.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from common import github_client, handle_numeric_id, merge_queue_status, read_agent_item, require_env, resolve_repository

if TYPE_CHECKING:
    from github.PullRequest import PullRequest

MARKER = "<!-- ci-doctor-mq-readd -->"


def already_readded(pull: "PullRequest") -> bool:
    """Return True if a previous CI Doctor re-add marker comment is present."""
    return any(MARKER in (comment.body or "") for comment in pull.get_issue_comments())


def skip_reason(pull: "PullRequest", pr_number: str, status: str) -> str | None:
    """Return a human-readable reason to skip re-adding, or None to proceed.

    ``status`` is the PR's **live** merge-queue membership resolved at this moment; 
    re-add only when it is
    ``not_in_queue`` so a status change during the investigation cannot cause a
    wrong or redundant enqueue.
    """
    if pull.merged:
        return f"PR #{pr_number} is already merged; nothing to re-add."
    if pull.state != "open":
        return f"PR #{pr_number} is not open (state={pull.state}); not re-adding."
    if pull.draft:
        return f"PR #{pr_number} is a draft; not re-adding."
    if status == "in_queue":
        return f"PR #{pr_number} is already in the merge queue; not re-adding."
    if status != "not_in_queue":
        return f"PR #{pr_number} merge-queue status is '{status}'; not re-adding (fail safe)."
    return None



def enqueue(pull: "PullRequest", pr_number: str, repository: str) -> None:
    """Re-add the PR to the merge queue via ``gh pr merge``.

    On a branch that requires a merge queue, ``gh pr merge`` adds the PR to the
    queue (using the queue's own configured merge method) instead of merging
    directly. ``--squash`` keeps the CLI non-interactive; ``--match-head-commit``
    guards against the head moving under us.
    """
    merge_command = ["gh", "pr", "merge", pr_number, "--repo", repository, "--squash"]
    if pull.head.sha:
        merge_command += ["--match-head-commit", pull.head.sha]
    result = subprocess.run(merge_command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Merge-queue enqueue failed: {result.stderr.strip()}")


def record_marker(pull: "PullRequest", reason: str) -> None:
    """Comment a marker so subsequent CI Doctor runs do not re-add again."""
    comment_body = (
        f"{MARKER}\n\n_CI Doctor re-added this pull request to the merge queue "
        f"after a transient failure (reason: {reason})._"
    )
    pull.create_issue_comment(comment_body)


def main() -> None:
    token = require_env("GH_TOKEN", "MERGE_QUEUE_TOKEN secret is not configured; cannot re-add to merge queue.")
    item = read_agent_item("readd_to_merge_queue")
    pr_number = handle_numeric_id(item.get("pr_number") or "", "pr_number")
    repository = resolve_repository(item.get("repository") or "")
    reason = item.get("reason") or ""

    print(f"Requested re-add of PR #{pr_number} in {repository} to merge queue (reason: {reason})")

    gh = github_client(token)
    repo = gh.get_repo(repository)
    pull = repo.get_pull(int(pr_number))

    if already_readded(pull):
        print(f"PR #{pr_number} already re-added by CI Doctor (marker comment found); skipping.")
        return

    status = merge_queue_status(gh, repo, pull)
    print(f"Live merge-queue status for PR #{pr_number}: {status}")

    reason_to_skip = skip_reason(pull, pr_number, status)
    if reason_to_skip:
        print(reason_to_skip)
        return

    enqueue(pull, pr_number, repository)
    print(f"Successfully requested re-add of PR #{pr_number} to the merge queue.")

    record_marker(pull, reason)
    print(f"Recorded re-add marker comment on PR #{pr_number}.")


if __name__ == "__main__":
    main()
