# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Re-add a PR after CI Doctor marks its merge-queue failure transient."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from common import (
    github_client,
    handle_numeric_id,
    merge_queue_status,
    pr_number_from_merge_queue_branch,
    require_env,
    resolve_repository,
)

if TYPE_CHECKING:
    from github.IssueComment import IssueComment
    from github.PullRequest import PullRequest
    from github.Repository import Repository
    from github.WorkflowRun import WorkflowRun


ANALYSIS_WORKFLOW_NAME = "CI Failure Doctor — Merge Queue"
BOT_LOGIN = "github-actions[bot]"
ELIGIBLE_MARKER_RE = re.compile(
    r"<!-- ci-doctor-mq-eligible pr=(?P<pr_number>\d+) "
    r"head_sha=(?P<head_sha>[0-9a-f]{40}) "
    r"source_run_id=(?P<source_run_id>\d+) "
    r"analysis_run_id=(?P<analysis_run_id>\d+) -->"
)
READD_MARKER = "<!-- ci-doctor-mq-readd -->"


@dataclass(frozen=True)
class EligibilityMarker:
    pr_number: str
    head_sha: str
    source_run_id: str
    analysis_run_id: str
    created_at: datetime


def parse_marker(body: str | None, created_at: datetime) -> EligibilityMarker | None:
    """Parse the machine-readable marker from a comment body."""
    match = ELIGIBLE_MARKER_RE.search(body or "")
    if not match:
        return None
    return EligibilityMarker(
        pr_number=match["pr_number"],
        head_sha=match["head_sha"],
        source_run_id=match["source_run_id"],
        analysis_run_id=match["analysis_run_id"],
        created_at=created_at,
    )


def find_eligibility_marker(
    comments: list[IssueComment], pr_number: int, head_sha: str
) -> EligibilityMarker | None:
    """Return the newest trusted marker for this PR's current head."""
    valid: list[EligibilityMarker] = []
    for comment in comments:
        user = getattr(comment, "user", None)
        if not user or getattr(user, "login", None) != BOT_LOGIN:
            continue
        marker = parse_marker(getattr(comment, "body", None), comment.created_at)
        if marker and marker.pr_number == str(pr_number) and marker.head_sha == head_sha:
            valid.append(marker)
    return max(valid, key=lambda marker: marker.created_at) if valid else None


def dequeue_reason(event: dict[str, Any]) -> str:
    """Read the reason field from a pull_request dequeued webhook payload."""
    pull_request = event.get("pull_request") or {}
    return str(event.get("reason") or pull_request.get("reason") or "")


def pr_number_from_event(event: dict[str, Any]) -> str:
    """Resolve the PR number from either the dequeue or analysis-completion event."""
    if event.get("action") == "dequeued":
        pull_request = event.get("pull_request") or {}
        return handle_numeric_id(str(pull_request.get("number") or ""), "pr_number")

    workflow_run = event.get("workflow_run") or {}
    if workflow_run.get("name") != ANALYSIS_WORKFLOW_NAME:
        raise SystemExit("workflow_run event did not originate from CI Doctor MQ.")
    if workflow_run.get("conclusion") not in ("success", "failure"):
        raise SystemExit("CI Doctor MQ did not complete normally; skipping re-queue.")
    branch = str(workflow_run.get("display_title") or "")
    number = pr_number_from_merge_queue_branch(branch)
    if not number:
        raise SystemExit("CI Doctor MQ run title does not contain a merge-queue PR branch.")
    return number


def _timeline_event_data(event: Any) -> tuple[str, datetime | None, str]:
    raw_data = getattr(event, "raw_data", {}) or {}
    name = str(getattr(event, "event", None) or raw_data.get("event") or "")
    created_at = getattr(event, "created_at", None)
    reason = str(getattr(event, "reason", None) or raw_data.get("reason") or "")
    return name, created_at, reason


def latest_queue_events(pull: PullRequest) -> tuple[tuple[str, datetime, str] | None, datetime | None]:
    """Return the latest add/remove event and latest enqueue timestamp."""
    queue_events: list[tuple[str, datetime, str]] = []
    latest_added: datetime | None = None
    for event in pull.as_issue().get_timeline():
        name, created_at, reason = _timeline_event_data(event)
        if name not in ("added_to_merge_queue", "removed_from_merge_queue") or created_at is None:
            continue
        queue_events.append((name, created_at, reason))
        if name == "added_to_merge_queue" and (latest_added is None or created_at > latest_added):
            latest_added = created_at
    latest = max(queue_events, key=lambda item: item[1]) if queue_events else None
    return latest, latest_added


def marker_is_current(marker: EligibilityMarker, latest_added: datetime | None) -> bool:
    """Reject markers posted before the current merge-queue admission."""
    return latest_added is not None and marker.created_at > latest_added


def marker_matches_analysis_run(marker: EligibilityMarker, workflow_run: dict[str, Any]) -> bool:
    """Ensure the completion fallback belongs to the analysis run that posted the marker."""
    return marker.analysis_run_id == str(workflow_run.get("id") or "")


def source_run_matches_pr(run: WorkflowRun, pr_number: str) -> bool:
    """Ensure the marked source run was a failed merge-group run for this PR."""
    return (
        run.event == "merge_group"
        and run.conclusion == "failure"
        and pr_number_from_merge_queue_branch(run.head_branch or "") == pr_number
    )


def _already_readded(pull: PullRequest) -> bool:
    """Return True when CI Doctor has previously re-added this PR."""
    return any(READD_MARKER in (comment.body or "") for comment in pull.get_issue_comments())


def _skip_reason(pull: PullRequest, pr_number: str) -> str | None:
    if pull.merged:
        return f"PR #{pr_number} is already merged; nothing to re-add."
    if pull.state != "open":
        return f"PR #{pr_number} is not open (state={pull.state}); not re-adding."
    if pull.draft:
        return f"PR #{pr_number} is a draft; not re-adding."
    return None


def readd_to_queue(
    pull: PullRequest,
    pr_number: str,
    repository: str,
    reason: str,
    merge_queue_token: str,
    gh: Any,
    repo: Repository,
) -> None:
    """Re-add the validated PR to the merge queue using the dedicated token."""
    if _already_readded(pull):
        print(f"PR #{pr_number} already carries the CI Doctor re-add marker; skipping.")
        return

    write_pull = github_client(merge_queue_token).get_repo(repository).get_pull(int(pr_number))
    skip_reason = _skip_reason(write_pull, pr_number)
    if skip_reason:
        print(skip_reason)
        return

    head_sha = pull.head.sha if pull.head else ""
    write_head_sha = write_pull.head.sha if write_pull.head else ""
    if not head_sha or not write_head_sha:
        raise SystemExit(f"PR #{pr_number} has no current head SHA; refusing to re-add.")
    if write_head_sha != head_sha:
        print(f"PR #{pr_number} head changed after validation; refusing to re-add.")
        return

    status = merge_queue_status(gh, repo, pull)
    if status != "not_in_queue":
        print(f"PR #{pr_number} is no longer confirmed outside the merge queue; skipping.")
        return

    # --match-head-commit prevents enqueueing a head that changed after validation.
    command = [
        "gh", "pr", "merge", pr_number, "--repo", repository, "--squash",
        "--match-head-commit", head_sha,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env={**os.environ, "GH_TOKEN": merge_queue_token}
    )
    if result.returncode != 0:
        raise SystemExit(f"Merge-queue enqueue failed: {result.stderr.strip()}")

    comment = (
        f"{READD_MARKER}\n\n"
        "### CI Doctor — Merge Queue Re-add\n\n"
        "CI Doctor re-added this PR after confirming that the merge queue removed it for "
        f"`failed_checks` and its current head was marked transient.\n\n"
        f"- **Reason**: {reason or 'transient failure'}\n"
    )
    write_pull.create_issue_comment(comment)
    _set_github_env("READDED_TO_MQ", "true")
    print(f"Successfully re-added PR #{pr_number} to the merge queue.")


def _set_github_env(key: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def process_event(event: dict[str, Any], gh: Any, repo: Repository, repository: str) -> None:
    """Validate the event, marker, source run, queue state, then re-add if safe."""
    is_dequeued = event.get("action") == "dequeued"
    event_reason = dequeue_reason(event)
    if is_dequeued and event_reason and event_reason != "failed_checks":
        print(f"Dequeued event reason is {event_reason!r}, not 'failed_checks'; skipping.")
        return

    if not is_dequeued:
        workflow_run = event.get("workflow_run") or {}
        if workflow_run.get("conclusion") not in ("success", "failure"):
            print("CI Doctor MQ did not complete normally; skipping.")
            return
        if not pr_number_from_merge_queue_branch(str(workflow_run.get("display_title") or "")):
            print("CI Doctor MQ run title does not contain a merge-queue PR branch; skipping.")
            return

    pr_number = pr_number_from_event(event)
    pull = repo.get_pull(int(pr_number))
    marker = find_eligibility_marker(list(pull.get_issue_comments()), pull.number, pull.head.sha or "")
    if not marker:
        print(f"No trusted transient-eligibility marker for PR #{pr_number}'s current head; skipping.")
        return

    if not is_dequeued:
        workflow_run = event["workflow_run"]
        if not marker_matches_analysis_run(marker, workflow_run):
            print("Eligibility marker does not match the completed CI Doctor MQ run; skipping.")
            return

    latest_event, latest_added = latest_queue_events(pull)
    if not latest_event or latest_event[0] != "removed_from_merge_queue" or latest_event[2] != "failed_checks":
        print("Latest merge-queue event is not a failed_checks removal; skipping.")
        return
    if not marker_is_current(marker, latest_added):
        print("Eligibility marker predates the latest merge-queue admission; skipping.")
        return

    source_run = repo.get_workflow_run(int(marker.source_run_id))
    if not source_run_matches_pr(source_run, pr_number):
        print("Eligibility marker source is not a failed merge-group run for this PR; skipping.")
        return

    status = merge_queue_status(gh, repo, pull)
    print(f"Live merge-queue status for PR #{pr_number}: {status}")
    if status != "not_in_queue":
        print(f"PR #{pr_number} is not confirmed outside the merge queue; skipping.")
        return

    merge_queue_token = require_env(
        "MERGE_QUEUE_TOKEN",
        "MERGE_QUEUE_TOKEN secret is not configured; cannot re-add to the merge queue.",
    )
    reason = "transient merge-queue failure"
    readd_to_queue(pull, pr_number, repository, reason, merge_queue_token, gh, repo)


def main() -> None:
    event_path = require_env("GITHUB_EVENT_PATH", "GITHUB_EVENT_PATH is not configured.")
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)

    repository = resolve_repository(os.environ.get("GITHUB_REPOSITORY", ""))
    gh = github_client(require_env("GH_TOKEN", "GITHUB_TOKEN is not configured."))
    repo = gh.get_repo(repository)
    process_event(event, gh, repo, repository)


if __name__ == "__main__":
    main()
