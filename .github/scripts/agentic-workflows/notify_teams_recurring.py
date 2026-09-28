# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Send a recurring merge-queue failure escalation alert to Microsoft Teams.

Used by the `notify-teams-recurring` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/notify-teams-recurring.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT and posts an Adaptive
Card to TEAMS_WEBHOOK_URL.
"""

from __future__ import annotations

from typing import Any

from common import post_adaptive_card, read_agent_item, require_env


def build_facts(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build the fixed three-fact FactSet for the recurring-failure card."""
    failed_workflow = item.get("failed_workflow") or ""
    pipeline_url = item.get("pipeline_url") or ""
    recent_count = item.get("recent_count") or ""
    return [
        {"title": "Workflow", "value": failed_workflow},
        {"title": "Pipeline", "value": f"[Latest run]({pipeline_url})"},
        {"title": "Hits (last 12 hrs)", "value": f"{recent_count} occurrences"},
    ]


def build_body(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the Adaptive Card body blocks for the recurring-failure alert."""
    title = item.get("title") or ""
    description = item.get("description") or ""
    affected_prs = item.get("affected_prs") or ""
    recent_runs = item.get("recent_run_urls") or ""
    return [
        {"type": "TextBlock", "text": f"\U0001f501 [MQ] Recurring Failure: {title}", "weight": "Bolder", "size": "Medium", "color": "Warning", "wrap": True},
        {"type": "FactSet", "facts": build_facts(item)},
        {"type": "TextBlock", "text": description, "wrap": True, "spacing": "Medium"},
        {"type": "TextBlock", "text": "### Affected PRs", "weight": "Bolder", "spacing": "Large", "separator": True},
        {"type": "TextBlock", "text": affected_prs, "wrap": True, "spacing": "Small"},
        {"type": "TextBlock", "text": "### Recent Failure Runs", "weight": "Bolder", "spacing": "Large", "separator": True},
        {"type": "TextBlock", "text": recent_runs, "wrap": True, "spacing": "Small"},
    ]


def main() -> None:
    webhook_url = require_env("TEAMS_WEBHOOK_URL", "TEAMS_WEBHOOK_URL secret is not configured")
    item = read_agent_item("notify_teams_recurring")
    post_adaptive_card(webhook_url, build_body(item))


if __name__ == "__main__":
    main()
