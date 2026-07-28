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
        - name: Send recurring failure escalation to Teams
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
          run: |
            set -euo pipefail

            if [ -z "${TEAMS_WEBHOOK_URL:-}" ]; then
              echo "TEAMS_WEBHOOK_URL secret is not configured" >&2
              exit 1
            fi

            if [ ! -f "${GH_AW_AGENT_OUTPUT:-}" ]; then
              echo "No agent output found at GH_AW_AGENT_OUTPUT" >&2
              exit 1
            fi

            ITEM=$(jq -c '[.items[] | select(.type == "notify_teams_recurring")] | last' "$GH_AW_AGENT_OUTPUT")
            if [ -z "$ITEM" ] || [ "$ITEM" = "null" ]; then
              echo "No notify_teams_recurring item present in agent output" >&2
              exit 1
            fi

            TITLE=$(echo "$ITEM"            | jq -r '.title // ""')
            FAILED_WORKFLOW=$(echo "$ITEM"  | jq -r '.failed_workflow // ""')
            PIPELINE_URL=$(echo "$ITEM"     | jq -r '.pipeline_url // ""')
            RECENT_COUNT=$(echo "$ITEM"     | jq -r '.recent_count // ""')
            DESCRIPTION=$(echo "$ITEM"      | jq -r '.description // ""')
            AFFECTED_PRS=$(echo "$ITEM"     | jq -r '.affected_prs // ""')
            RECENT_RUNS=$(echo "$ITEM"      | jq -r '.recent_run_urls // ""')

            FACTS=$(jq -nc \
              --arg failed_workflow "$FAILED_WORKFLOW" \
              --arg pipeline_url    "$PIPELINE_URL" \
              --arg recent_count    "$RECENT_COUNT" '
                [
                  { title: "Workflow",           value: $failed_workflow },
                  { title: "Pipeline",           value: ("[Latest run](" + $pipeline_url + ")") },
                  { title: "Hits (last 12 hrs)", value: ($recent_count + " occurrences") }
                ]')

            PAYLOAD=$(jq -nc \
              --arg title "$TITLE" \
              --arg description "$DESCRIPTION" \
              --arg affected_prs "$AFFECTED_PRS" \
              --arg recent_runs "$RECENT_RUNS" \
              --argjson facts "$FACTS" '
                {
                  type: "message",
                  attachments: [{
                    contentType: "application/vnd.microsoft.card.adaptive",
                    content: {
                      "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                      type: "AdaptiveCard",
                      version: "1.4",
                      body: [
                        { type: "TextBlock", text: ("\ud83d\udd01 [MQ] Recurring Failure: " + $title), weight: "Bolder", size: "Medium", color: "Warning", wrap: true },
                        { type: "FactSet", facts: $facts },
                        { type: "TextBlock", text: $description, wrap: true, spacing: "Medium" },
                        { type: "TextBlock", text: "### Affected PRs", weight: "Bolder", spacing: "Large", separator: true },
                        { type: "TextBlock", text: $affected_prs, wrap: true, spacing: "Small" },
                        { type: "TextBlock", text: "### Recent Failure Runs", weight: "Bolder", spacing: "Large", separator: true },
                        { type: "TextBlock", text: $recent_runs, wrap: true, spacing: "Small" }
                      ]
                    }
                  }]
                }')

            curl -sS --fail-with-body \
              -H "Content-Type: application/json" \
              -d "$PAYLOAD" \
              "$TEAMS_WEBHOOK_URL"
---

# CI Doctor MQ — Recurring Teams Notification Job

Shared definition of the `notify-teams-recurring` custom safe-output job used by
the CI Doctor Merge Queue workflow. Import it via `imports:` in the consuming
workflow's frontmatter.
