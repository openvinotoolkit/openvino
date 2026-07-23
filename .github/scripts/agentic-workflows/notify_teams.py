#!/usr/bin/env python3
"""Send a CI failure investigation summary to Microsoft Teams.

Used by the `notify-teams` custom safe-output job of the CI Doctor MQ workflow
(.github/workflows/shared/agentic-workflows/notify-teams.md). Reads the agent
output referenced by GH_AW_AGENT_OUTPUT, persists the statistics database as a
workflow artifact, and posts an Adaptive Card to TEAMS_WEBHOOK_URL.
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

items = [it for it in payload_items if it.get("type") == "notify_teams"]
if not items:
    sys.exit("No notify_teams item present in agent output")
item = items[-1]

title = item.get("title") or ""
failed_workflow = item.get("failed_workflow") or ""
pipeline_url = item.get("pipeline_url") or ""
description = item.get("description") or ""
pr_number = item.get("pr_number") or ""
pr_url = item.get("pr_url") or ""
author = item.get("author") or ""
db_entries = item.get("db_entries") or ""
occurrences = item.get("occurrence_count") or ""
statistics = item.get("statistics") or ""
statistics_json = item.get("statistics_json") or ""

# Persist the full statistics database as a workflow artifact for offline review.
stats_dir = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "ci-doctor-mq-stats")
os.makedirs(stats_dir, exist_ok=True)
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
if statistics:
    with open(os.path.join(stats_dir, "ci-doctor-mq-statistics.md"), "w", encoding="utf-8") as handle:
        handle.write(statistics + "\n")

github_output = os.environ.get("GITHUB_OUTPUT")
if github_output:
    with open(github_output, "a", encoding="utf-8") as handle:
        handle.write(f"stats_dir={stats_dir}\n")

# Build Adaptive Card facts conditionally (only include PR/author when present).
facts = []
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

body = [
    {"type": "TextBlock", "text": f"\U0001f534 [MQ] {title}", "weight": "Bolder", "size": "Medium", "color": "Attention", "wrap": True},
    {"type": "FactSet", "facts": facts},
    {"type": "TextBlock", "text": description, "wrap": True, "spacing": "Medium"},
]
if statistics:
    body.append({"type": "TextBlock", "text": "Pattern Database Statistics", "weight": "Bolder", "size": "Medium", "spacing": "Large", "separator": True})
    body.append({"type": "TextBlock", "text": statistics, "wrap": True, "spacing": "Small"})

payload = {
    "type": "message",
    "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
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
