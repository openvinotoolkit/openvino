# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the CI Doctor agentic-workflow scripts.

Imported by the sibling scripts in this directory. They run as
``python .github/scripts/agentic-workflows/<name>.py``, so this module's
directory is on ``sys.path`` and a plain ``import common`` resolves here.

Only the standard library is imported at module scope; PyGithub is imported
lazily inside :func:`github_client` so scripts that do not need it (the Teams
notifiers) can run without PyGithub installed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from github import Github
    from github.PullRequest import PullRequest
    from github.Repository import Repository

_NUMERIC_RE = re.compile(r"[0-9]+")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")

# Merge-queue branches look like: gh-readonly-queue/<base_branch>/pr-<number>-<sha>
MERGE_QUEUE_PR_RE = re.compile(r"/pr-(\d+)-[0-9a-f]+$")

# Live merge-queue membership for a base branch. `mergeQueue` is null when the
# branch has no queue configured, which we treat as indeterminate.
_MERGE_QUEUE_QUERY = """
query($owner: String!, $repo: String!, $branch: String!) {
  repository(owner: $owner, name: $repo) {
    mergeQueue(branch: $branch) {
      entries(first: 100) { nodes { pullRequest { number } } }
    }
  }
}
"""

# Timeline events that flip merge-queue membership; latest one wins.
_MERGE_QUEUE_TIMELINE_EVENTS = {
    "added_to_merge_queue": True,
    "removed_from_merge_queue": False,
}

MergeQueueStatus = Literal["in_queue", "not_in_queue", "merged", "unknown"]



def read_agent_item(item_type: str) -> dict[str, Any]:
    """Return the last agent-output item of ``item_type`` (exits if absent).

    The agent output path is taken from ``GH_AW_AGENT_OUTPUT``.
    """
    agent_output = os.environ.get("GH_AW_AGENT_OUTPUT", "")
    if not agent_output or not os.path.isfile(agent_output):
        sys.exit("No agent output found at GH_AW_AGENT_OUTPUT")

    with open(agent_output, encoding="utf-8") as handle:
        items = json.load(handle).get("items", [])

    matching = [item for item in items if item.get("type") == item_type]
    if not matching:
        sys.exit(f"No {item_type} item present in agent output")
    return matching[-1]


def require_env(name: str, message: str | None = None) -> str:
    """Return a non-empty environment variable ``name`` or exit with a message."""
    value = os.environ.get(name, "")
    if not value:
        sys.exit(message or f"{name} is not configured")
    return value


def handle_numeric_id(value: str, label: str) -> str:
    """Validate a purely numeric id (e.g. run_id / pr_number) to avoid injection."""
    if not _NUMERIC_RE.fullmatch(value):
        sys.exit(f"{label} must be a numeric string, got: '{value}'")
    return value


def resolve_repository(repository: str) -> str:
    """Fall back to ``$GITHUB_REPOSITORY`` and validate the ``owner/repo`` form."""
    if not repository or repository == "not_found":
        repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not _REPOSITORY_RE.fullmatch(repository):
        sys.exit(f"repository must be in owner/repo format, got: '{repository}'")
    return repository


def github_client(token: str) -> "Github":
    """Build an authenticated PyGithub client (PyGithub imported lazily)."""
    from github import Auth, Github

    return Github(auth=Auth.Token(token))


def _merge_queue_via_graphql(gh: "Github", owner: str, repo: str, branch: str, pr_number: int) -> bool | None:
    """Return whether the PR is a live merge-queue entry, or None if indeterminate."""
    if not branch:
        return None
    try:
        _, data = gh.requester.requestJsonAndCheck(
            "POST",
            "/graphql",
            input={"query": _MERGE_QUEUE_QUERY, "variables": {"owner": owner, "repo": repo, "branch": branch}},
        )
    except Exception as error:
        print(f"GraphQL merge-queue lookup failed: {error}")
        return None
    if data.get("errors"):
        print(f"GraphQL merge-queue lookup returned errors: {data['errors']}")
        return None
    merge_queue = data.get("data", {}).get("repository", {}).get("mergeQueue")
    if merge_queue is None:
        return None
    nodes = (merge_queue.get("entries") or {}).get("nodes") or []
    return any((node.get("pullRequest") or {}).get("number") == pr_number for node in nodes)


def _merge_queue_via_timeline(pull: "PullRequest") -> bool | None:
    """Return membership from the latest add/remove timeline event, or None if unknown."""
    latest_state: bool | None = None
    latest_time = None
    try:
        for event in pull.as_issue().get_timeline():
            state = _MERGE_QUEUE_TIMELINE_EVENTS.get(getattr(event, "event", None))
            created = getattr(event, "created_at", None)
            if state is None or created is None:
                continue
            if latest_time is None or created >= latest_time:
                latest_time, latest_state = created, state
    except Exception as error:
        print(f"Timeline merge-queue lookup failed: {error}")
        return None
    return latest_state


def merge_queue_status(gh: "Github", repo: "Repository", pull: "PullRequest") -> MergeQueueStatus:
    """Resolve a PR's **current** merge-queue membership deterministically.

    Returns one of ``in_queue`` / ``not_in_queue`` / ``merged`` / ``unknown``.
    Prefers the live GraphQL ``mergeQueue`` entries; falls back to the PR's
    add/remove timeline events; reports ``unknown`` only when both are
    indeterminate so callers can fail safe. Because it queries live state, call
    it at the moment a decision is made (e.g. inside a safe-output job right
    before mutating), not from a stale earlier snapshot.
    """
    if pull.merged:
        return "merged"
    if (pull.state or "").lower() == "closed":
        return "not_in_queue"

    owner, _, name = repo.full_name.partition("/")
    branch = pull.base.ref if pull.base else ""

    membership = _merge_queue_via_graphql(gh, owner, name, branch, pull.number)
    if membership is None:
        membership = _merge_queue_via_timeline(pull)
    if membership is None:
        return "unknown"
    return "in_queue" if membership else "not_in_queue"


def pr_number_from_merge_queue_branch(head_branch: str) -> str:
    """Extract the PR number from a ``gh-readonly-queue/...`` head branch, or ''."""
    match = MERGE_QUEUE_PR_RE.search(head_branch or "")
    return match.group(1) if match else ""



def post_adaptive_card(webhook_url: str, body: list[dict[str, Any]]) -> None:
    """POST an Adaptive Card ``body`` to a Microsoft Teams incoming webhook."""
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }
    if urllib.parse.urlsplit(webhook_url).scheme != "https":
        sys.exit(f"Teams webhook URL must use https, got: '{webhook_url}'")
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:  # nosec B310 - scheme restricted to https above
            print(f"Teams webhook responded with HTTP {response.status}: {response.read().decode('utf-8', 'replace')}")
    except urllib.error.HTTPError as error:
        sys.exit(f"Teams webhook failed with HTTP {error.code}: {error.read().decode('utf-8', 'replace')}")
