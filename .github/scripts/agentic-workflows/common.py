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
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from github import Github

_NUMERIC_RE = re.compile(r"[0-9]+")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")


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
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(f"Teams webhook responded with HTTP {response.status}: {response.read().decode('utf-8', 'replace')}")
    except urllib.error.HTTPError as error:
        sys.exit(f"Teams webhook failed with HTTP {error.code}: {error.read().decode('utf-8', 'replace')}")
