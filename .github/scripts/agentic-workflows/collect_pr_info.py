# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre-collect pull-request metadata for the CI Doctor workflows.

Used by the shared pre-agent step
(.github/workflows/shared/agentic-workflows/collect-pr-info.md). Auto-detects its
mode from the environment (mirroring ``download_failure_logs.py``):
  - run mode (RUN_ID set):    a merge-queue (``merge_group``) run carries no
    direct PR context, so the PR number is recovered from the merge-queue head
    branch (``gh-readonly-queue/<base>/pr-<number>-<sha>``), falling back to the
    run's associated ``pull_requests``.
  - pr mode  (PR_NUMBER set): the PR number is already known (e.g. a
    ``/ci-doctor`` pull-request comment), so the PR is fetched directly.

Resolving it deterministically before the session and writing it to
``/tmp/gh-aw/agent/ci-doctor/pr-info.json`` (plus a human-readable
``pr-info.txt``) lets the agent populate the Teams ``PR`` / ``Author`` fields and
scope its source inspection to the PR diff.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

from common import github_client

if TYPE_CHECKING:
    from github.PullRequest import PullRequest
    from github.Repository import Repository
    from github.WorkflowRun import WorkflowRun

OUTPUT_DIR = "/tmp/gh-aw/agent/ci-doctor"
PR_INFO_JSON = os.path.join(OUTPUT_DIR, "pr-info.json")
PR_INFO_TXT = os.path.join(OUTPUT_DIR, "pr-info.txt")

# Merge-queue branches look like: gh-readonly-queue/<base_branch>/pr-<number>-<sha>
MERGE_QUEUE_PR_RE = re.compile(r"/pr-(\d+)-[0-9a-f]+$")

# Cap the number of changed files and the PR body length written to disk so the
# artifact stays small and the agent context is not flooded.
MAX_CHANGED_FILES = 100
MAX_BODY_CHARS = 4000


def pr_number_from_run(workflow_run: "WorkflowRun") -> str:
    """Resolve the PR number behind a merge-queue run.

    Prefers the ``gh-readonly-queue/<base>/pr-<number>-<sha>`` head branch used by
    the merge queue; falls back to the run's associated ``pull_requests`` list.
    Returns an empty string when no PR can be identified.
    """
    head_branch = workflow_run.head_branch or ""
    match = MERGE_QUEUE_PR_RE.search(head_branch)
    if match:
        return match.group(1)

    try:
        pulls = workflow_run.pull_requests or []
    except Exception:
        pulls = []
    if pulls:
        return str(pulls[0].number)

    return ""


def changed_files(pull: "PullRequest") -> list[str]:
    """Return up to ``MAX_CHANGED_FILES`` changed file paths for the PR."""
    files: list[str] = []
    try:
        for changed in pull.get_files():
            files.append(changed.filename)
            if len(files) >= MAX_CHANGED_FILES:
                break
    except Exception:
        return files
    return files


def build_pr_info(pull: "PullRequest") -> dict[str, Any]:
    """Assemble the serialisable PR metadata dictionary."""
    body = pull.body or ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n...(truncated)..."

    author = pull.user.login if pull.user else ""
    author_url = pull.user.html_url if pull.user else ""

    return {
        "pr_number": str(pull.number),
        "pr_url": pull.html_url,
        "title": pull.title or "",
        "author": author,
        "author_url": author_url,
        "base_branch": pull.base.ref if pull.base else "",
        "head_ref": pull.head.ref if pull.head else "",
        "head_sha": pull.head.sha if pull.head else "",
        "state": pull.state or "",
        "draft": bool(pull.draft),
        "merged": bool(pull.merged),
        "labels": [label.name for label in pull.labels],
        "changed_files": changed_files(pull),
        "body": body,
    }


def write_outputs(pr_info: dict[str, Any] | None) -> None:
    """Write the JSON and human-readable PR-info artifacts to ``OUTPUT_DIR``."""
    with open(PR_INFO_JSON, "w", encoding="utf-8") as handle:
        json.dump(pr_info or {}, handle, indent=2)

    with open(PR_INFO_TXT, "w", encoding="utf-8") as handle:
        if not pr_info:
            handle.write("No pull request could be associated with this merge-queue run.\n")
            handle.write("Leave the Teams PR / Author fields unset (use 'not_found').\n")
            return
        handle.write("=== CI Doctor: Pull Request Info ===\n")
        handle.write(f"PR number:     {pr_info['pr_number']}\n")
        handle.write(f"PR URL:        {pr_info['pr_url']}\n")
        handle.write(f"Title:         {pr_info['title']}\n")
        handle.write(f"Author:        {pr_info['author']}\n")
        handle.write(f"Author URL:    {pr_info['author_url']}\n")
        handle.write(f"Base branch:   {pr_info['base_branch']}\n")
        handle.write(f"Head ref:      {pr_info['head_ref']}\n")
        handle.write(f"Head SHA:      {pr_info['head_sha']}\n")
        handle.write(f"State:         {pr_info['state']} (draft={pr_info['draft']}, merged={pr_info['merged']})\n")
        handle.write(f"Labels:        {', '.join(pr_info['labels']) or '(none)'}\n")
        handle.write("\n")
        handle.write(f"Changed files ({len(pr_info['changed_files'])}, capped at {MAX_CHANGED_FILES}):\n")
        for path in pr_info["changed_files"]:
            handle.write(f"  {path}\n")


def persist_pull(repo: "Repository", pr_number: str) -> None:
    """Fetch PR ``pr_number`` and persist its metadata (or the no-PR fallback)."""
    try:
        pull = repo.get_pull(int(pr_number))
    except Exception as error:
        print(f"Could not fetch PR #{pr_number}: {error}")
        write_outputs(None)
        return

    pr_info = build_pr_info(pull)
    write_outputs(pr_info)
    print(f"Wrote PR info to {PR_INFO_JSON} and {PR_INFO_TXT}")
    print(f"  PR #{pr_info['pr_number']} by @{pr_info['author']}: {pr_info['title']}")


def run_mode(repo: "Repository", run_id: str) -> None:
    """Resolve and persist the PR metadata behind a merge-queue run."""
    print(f"=== CI Doctor: Collecting PR info for run {run_id} ===")

    try:
        workflow_run = repo.get_workflow_run(int(run_id))
    except Exception as error:
        print(f"Could not fetch workflow run {run_id}: {error}")
        write_outputs(None)
        return

    pr_number = pr_number_from_run(workflow_run)
    if not pr_number:
        print(f"Could not identify a PR for run {run_id} (head branch: {workflow_run.head_branch!r})")
        write_outputs(None)
        return
    print(f"Resolved PR #{pr_number} from run {run_id}")

    persist_pull(repo, pr_number)


def pr_mode(repo: "Repository", pr_number: str) -> None:
    """Persist the PR metadata for an already-known PR number."""
    print(f"=== CI Doctor: Collecting PR info for PR #{pr_number} ===")
    persist_pull(repo, pr_number)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    repo_name = os.environ.get("REPO", "")
    run_id = (os.environ.get("RUN_ID") or "").strip()
    pr_number = (os.environ.get("PR_NUMBER") or "").strip()
    token = os.environ.get("GH_TOKEN", "")

    if not run_id and not pr_number:
        print("Neither RUN_ID nor PR_NUMBER is set; no PR to collect info for.")
        write_outputs(None)
        return

    repo = github_client(token).get_repo(repo_name)

    if run_id:
        run_mode(repo, run_id)
    else:
        pr_mode(repo, pr_number)


if __name__ == "__main__":
    main()
