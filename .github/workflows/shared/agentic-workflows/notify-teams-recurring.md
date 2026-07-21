---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Sends a recurring merge-queue failure escalation alert to Microsoft Teams.
safe-outputs:
  jobs:
    notify-teams-recurring:
      description: "Send a recurring merge-queue failure escalation alert to Microsoft Teams. Call this ONLY when the same failure pattern has 3 or more occurrences in the last 12 hours. Do NOT call this for every failure."
      runs-on: ubuntu-latest
      output: "Recurring failure escalation sent to Microsoft Teams."
      permissions:
        contents: read
      inputs:
        title:
          description: "Short, searchable description of the recurring failure pattern (same as notify_teams.title)."
          required: true
          type: string
        failed_workflow:
          description: "Name of the GitHub Actions workflow with the recurring failure."
          required: true
          type: string
        pipeline_url:
          description: "URL of the current (latest) failed workflow run."
          required: true
          type: string
        recent_count:
          description: "Number of times this failure pattern has occurred in the last 12 hours, including the current run. Report as a positive integer encoded as a string (e.g., '3', '5')."
          required: true
          type: string
        description:
          description: "Concise markdown gist of the recurring problem: what keeps failing, suspected root cause, and recommended escalation actions. Use Teams-safe markdown only (no raw HTML)."
          required: true
          type: string
        affected_prs:
          description: "Markdown-formatted list of affected PR numbers/links from the merge queue that hit this failure in the last 12 hours. One PR per line, e.g. '- [#1234](https://github.com/org/repo/pull/1234)'. Include up to 10 most recent PRs."
          required: true
          type: string
        recent_run_urls:
          description: "Markdown-formatted list of workflow run URLs that exhibited this failure in the last 12 hours. One URL per line, e.g. '- [Run 12345](https://github.com/org/repo/actions/runs/12345)'. Include up to 10 most recent runs."
          required: true
          type: string
      steps:
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Send recurring failure escalation to Teams
          shell: python
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
          run: |
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
---

# CI Doctor MQ — Recurring Teams Notification Job

Shared definition of the `notify-teams-recurring` custom safe-output job used by
the CI Doctor Merge Queue workflow. Import it via `imports:` in the consuming
workflow's frontmatter.
