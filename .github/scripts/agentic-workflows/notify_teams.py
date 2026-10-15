# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Send a CI failure investigation summary to Microsoft Teams.

Used by the `notify-teams` custom safe-output job of the CI Doctor MQ workflow
(.github/workflows/shared/agentic-workflows/notify-teams.md). Reads the agent
output referenced by GH_AW_AGENT_OUTPUT, persists the statistics database as a
workflow artifact, and posts an Adaptive Card to TEAMS_WEBHOOK_URL.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from common import post_adaptive_card, read_agent_item, require_env


def persist_statistics(item: dict[str, Any]) -> str:
    """Persist the statistics database as workflow-artifact files; return its dir."""
    stats_dir = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "ci-doctor-mq-stats")
    os.makedirs(stats_dir, exist_ok=True)

    statistics_json = item.get("statistics_json") or ""
    if statistics_json:
        stats_json_path = os.path.join(stats_dir, "ci-doctor-mq-statistics.json")
        try:
            # Validate and pretty-print; fall back to raw on parse error.
            parsed = json.loads(statistics_json)
            with open(stats_json_path, "w", encoding="utf-8") as handle:
                json.dump(parsed, handle, indent=2)
            print(f"Wrote validated statistics JSON ({os.path.getsize(stats_json_path)} bytes)")
        except json.JSONDecodeError:
            print("Warning: statistics_json failed JSON parse; storing raw payload", file=sys.stderr)
            with open(stats_json_path, "w", encoding="utf-8") as handle:
                handle.write(statistics_json)

    statistics = item.get("statistics") or ""
    if statistics:
        with open(os.path.join(stats_dir, "ci-doctor-mq-statistics.md"), "w", encoding="utf-8") as handle:
            handle.write(statistics + "\n")
    return stats_dir


def record_step_output(name: str, value: str) -> None:
    """Append ``name=value`` to the GitHub Actions step output file, when set."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def build_facts(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build the Adaptive Card FactSet, including only populated fields."""
    facts: list[dict[str, str]] = []
    failed_workflow = item.get("failed_workflow") or ""
    pipeline_url = item.get("pipeline_url") or ""
    pr_number = item.get("pr_number") or ""
    pr_url = item.get("pr_url") or ""
    author = item.get("author") or ""
    occurrences = item.get("occurrence_count") or ""
    db_entries = item.get("db_entries") or ""

    if failed_workflow:
        facts.append({"title": "Workflow", "value": failed_workflow})
    if pipeline_url:
        facts.append({"title": "Pipeline", "value": f"[Open run]({pipeline_url})"})
    if pr_number:
        pr_value = f"[#{pr_number}]({pr_url})" if pr_url else f"#{pr_number}"
        facts.append({"title": "PR", "value": pr_value})
    if author:
        facts.append({"title": "Author", "value": f"@{author}"})
    if occurrences:
        facts.append({"title": "Occurrences", "value": f"{occurrences}\u00d7"})
    if db_entries:
        facts.append({"title": "DB entries", "value": db_entries})
    return facts


def build_body(item: dict[str, Any], facts: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build the Adaptive Card body blocks for the investigation notification."""
    title = item.get("title") or ""
    description = item.get("description") or ""
    statistics = item.get("statistics") or ""

    body: list[dict[str, Any]] = [
        {"type": "TextBlock", "text": f"\U0001f534 [MQ] {title}", "weight": "Bolder", "size": "Medium", "color": "Attention", "wrap": True},
        {"type": "FactSet", "facts": facts},
        {"type": "TextBlock", "text": description, "wrap": True, "spacing": "Medium"},
    ]
    if statistics:
        body.append({"type": "TextBlock", "text": "Pattern Database Statistics", "weight": "Bolder", "size": "Medium", "spacing": "Large", "separator": True})
        body.append({"type": "TextBlock", "text": statistics, "wrap": True, "spacing": "Small"})
    return body


def main() -> None:
    webhook_url = require_env("TEAMS_WEBHOOK_URL", "TEAMS_WEBHOOK_URL secret is not configured")
    item = read_agent_item("notify_teams")

    # Persist the full statistics database as a workflow artifact for offline review.
    stats_dir = persist_statistics(item)
    record_step_output("stats_dir", stats_dir)

    facts = build_facts(item)
    body = build_body(item, facts)
    post_adaptive_card(webhook_url, body)


if __name__ == "__main__":
    main()
