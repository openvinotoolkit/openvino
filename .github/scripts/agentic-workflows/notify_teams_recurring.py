#!/usr/bin/env python3
"""Send a recurring merge-queue failure escalation alert to Microsoft Teams.

Used by the `notify-teams-recurring` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/notify-teams-recurring.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT and posts an Adaptive
Card to TEAMS_WEBHOOK_URL.
"""

import json
import os
import sys
import urllib.error
import urllib.request

webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
if not webhook_url:
    sys.exit("TEAMS_WEBHOOK_URL secret is not configured")

agent_output = os.environ.get("GH_AW_AGENT_OUTPUT", "")
if not agent_output or not os.path.isfile(agent_output):
    sys.exit("No agent output found at GH_AW_AGENT_OUTPUT")

with open(agent_output, encoding="utf-8") as handle:
    payload_items = json.load(handle).get("items", [])

items = [it for it in payload_items if it.get("type") == "notify_teams_recurring"]
if not items:
    sys.exit("No notify_teams_recurring item present in agent output")
item = items[-1]

title = item.get("title") or ""
failed_workflow = item.get("failed_workflow") or ""
pipeline_url = item.get("pipeline_url") or ""
recent_count = item.get("recent_count") or ""
description = item.get("description") or ""
affected_prs = item.get("affected_prs") or ""
recent_runs = item.get("recent_run_urls") or ""

facts = [
    {"title": "Workflow", "value": failed_workflow},
    {"title": "Pipeline", "value": f"[Latest run]({pipeline_url})"},
    {"title": "Hits (last 12 hrs)", "value": f"{recent_count} occurrences"},
]

payload = {
    "type": "message",
    "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": f"\U0001f501 [MQ] Recurring Failure: {title}", "weight": "Bolder", "size": "Medium", "color": "Warning", "wrap": True},
                {"type": "FactSet", "facts": facts},
                {"type": "TextBlock", "text": description, "wrap": True, "spacing": "Medium"},
                {"type": "TextBlock", "text": "### Affected PRs", "weight": "Bolder", "spacing": "Large", "separator": True},
                {"type": "TextBlock", "text": affected_prs, "wrap": True, "spacing": "Small"},
                {"type": "TextBlock", "text": "### Recent Failure Runs", "weight": "Bolder", "spacing": "Large", "separator": True},
                {"type": "TextBlock", "text": recent_runs, "wrap": True, "spacing": "Small"},
            ],
        },
    }],
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
