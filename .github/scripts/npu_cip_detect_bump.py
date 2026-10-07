# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detect NPU CiP checksum bumps and prepare GitHub Actions outputs for notification.

Used by the "Detect CiP value-level change" step of the NPU CiP Bump Notification
workflow (.github/workflows/npu_cip_bump_notification.yml). Fetches
download_compiler_libs.cmake at two revisions via the GitHub API, compares tracked
PLUGIN_COMPILER_* checksum variables, writes body.md for PR comments, and sets
step outputs consumed by npu_cip_notify_teams.py and the PR comment step.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ZERO_SHA = "0" * 40
SET_PATTERN = re.compile(r"set\(([^)\s]+)\s+([^)]*)\)")


def append_step_summary(text: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)


def set_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as handle:
        if "\n" in value:
            delimiter = f"{name}_EOF"
            while delimiter in value:
                delimiter += "_"
            handle.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
        else:
            handle.write(f"{name}={value}\n")


def gh_api_json(path: str) -> object | None:
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def fetch_file(repo: str, file_path: str, ref: str) -> str:
    payload = gh_api_json(f"repos/{repo}/contents/{file_path}?ref={ref}")
    if not isinstance(payload, dict) or "content" not in payload:
        sys.exit(f"Failed to fetch {file_path} at ref {ref}")
    raw = payload["content"].replace("\n", "")
    return base64.b64decode(raw).decode("utf-8")


def get_val(cmake_text: str, variable: str) -> str:
    prefix = f"set({variable}"
    for match in SET_PATTERN.finditer(cmake_text):
        if match.group(1) == variable:
            return re.sub(r"\s+", "", match.group(2))
    return ""


def resolve_before_after() -> tuple[str, str]:
    event_name = os.environ.get("EVENT_NAME", "")
    if event_name == "workflow_dispatch":
        return os.environ.get("INPUT_BEFORE_SHA", ""), os.environ.get("INPUT_AFTER_SHA", "")
    return os.environ.get("PUSH_BEFORE_SHA", ""), os.environ.get("PUSH_AFTER_SHA", "")


def build_mentions(logins: str) -> str:
    parts = []
    for login in logins.replace(",", " ").split():
        if login:
            parts.append(f"@{login}")
    return " ".join(parts) + (" " if parts else "")


def main() -> None:
    file_path = os.environ.get("FILE_PATH", "")
    tracked_raw = os.environ.get("TRACKED_VARS", "")
    tracked_vars = tracked_raw.split()
    repo = os.environ["GITHUB_REPOSITORY"]
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")

    before, after = resolve_before_after()
    print(f"Comparing {file_path} between {before} and {after}")

    if not before or before == ZERO_SHA:
        message = "No usable base revision (branch creation or force-push); skipping."
        print(message)
        append_step_summary(message + "\n")
        set_output("changed", "false")
        return

    before_text = fetch_file(repo, file_path, before)
    after_text = fetch_file(repo, file_path, after)

    changed = False
    anomaly = False
    parsed_any_after = False
    rows: list[str] = []
    changes_lines: list[str] = []

    for variable in tracked_vars:
        old = get_val(before_text, variable)
        new = get_val(after_text, variable)
        if new:
            parsed_any_after = True
        if old != new:
            changed = True
            rows.append(f"| `{variable}` | `{old or '<none>'}` | `{new or '<none>'}` |")
            changes_lines.append(f"- {variable}: {old or '<none>'} -> {new or '<none>'}")

    if not parsed_any_after:
        anomaly = True
        changed = True

    set_output("changed", "true" if changed else "false")
    if not changed:
        message = "No value-level change in tracked NPU CiP variables; skipping notification."
        print(message)
        append_step_summary(message + "\n")
        return

    now = datetime.now(tz=ZoneInfo("Europe/Warsaw")).strftime("%H:%M %d/%m/%Y %Z")
    commit_url = f"{server_url}/{repo}/commit/{after}"
    branch = ref_name

    author = ""
    commit_payload = gh_api_json(f"repos/{repo}/commits/{after}")
    if isinstance(commit_payload, dict):
        commit = commit_payload.get("commit") or {}
        if isinstance(commit, dict):
            author_info = commit.get("author") or {}
            if isinstance(author_info, dict):
                author = author_info.get("name") or ""

    pr_info = ""
    pr_number = ""
    pulls = gh_api_json(f"repos/{repo}/commits/{after}/pulls")
    if isinstance(pulls, list) and pulls:
        first = pulls[0]
        if isinstance(first, dict) and first.get("number") is not None:
            pr_number = str(first["number"])
            title = first.get("title") or ""
            pr_info = f"#{pr_number} {title}".strip()

    pr_url = f"{server_url}/{repo}/pull/{pr_number}" if pr_number else ""
    mentions = build_mentions(os.environ.get("NPU_CIP_NOTIFY_LOGINS", ""))

    body_lines = [
        "### 🔔 NPU CiP bump detected",
        "",
    ]
    if mentions.strip():
        body_lines.append(mentions.rstrip())
        body_lines.append("")
    body_lines.extend([
        "| Field | Value |",
        "| --- | --- |",
        f"| Date | {now} |",
        f"| Branch | `{branch}` |",
        f"| Commit | {commit_url} |",
    ])
    if author:
        body_lines.append(f"| Author | {author} |")
    if pr_info:
        body_lines.append(f"| Pull request | {pr_info} |")
    body_lines.append("")
    if anomaly:
        body_lines.extend([
            "> ⚠️ The tracked `PLUGIN_COMPILER_*` variables could not be parsed in the new",
            "> revision. They may have been renamed — please verify this notification manually.",
            "",
        ])
    if rows:
        body_lines.extend([
            "| Variable | Old | New |",
            "| --- | --- | --- |",
            *rows,
        ])
    body_md = "\n".join(body_lines) + "\n"

    append_step_summary(body_md)
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    (workspace / "body.md").write_text(body_md, encoding="utf-8")

    changes_text = "\n".join(changes_lines) if changes_lines else "<variables could not be parsed>"
    set_output("date", now)
    set_output("branch", branch)
    set_output("commit_url", commit_url)
    set_output("author", author)
    set_output("pr_info", pr_info)
    set_output("pr_number", pr_number)
    set_output("pr_url", pr_url)
    set_output("changes", changes_text)


if __name__ == "__main__":
    main()
