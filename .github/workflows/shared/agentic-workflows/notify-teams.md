---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Sends a CI failure investigation summary to Microsoft Teams.
safe-outputs:
  jobs:
    notify-teams:
      description: "Send a CI failure investigation summary to Microsoft Teams. Call this exactly once at the end of the investigation with a concise title and a thorough description of the failure."
      runs-on: ubuntu-latest
      output: "Notification sent to Microsoft Teams."
      permissions:
        contents: read
      inputs:
        title:
          description: "Short, searchable description of the failure (e.g. 'smoke_Bucketize tests fail on comparison'). No PR/run numbers."
          required: true
          type: string
        failed_workflow:
          description: "Name of the GitHub Actions workflow that failed (as reported by `get_workflow_run`, e.g. 'Linux (Ubuntu 22.04, Python 3.11)'). Do NOT pass the CI Doctor MQ workflow name."
          required: true
          type: string
        pipeline_url:
          description: "URL of the failed GitHub Actions workflow run."
          required: true
          type: string
        description:
          description: "Thorough markdown description of the problem: root cause, failed jobs, key error messages, and recommended actions."
          required: true
          type: string
        pr_number:
          description: "Pull request number if the failure is associated with a PR in the merge queue. Omit otherwise."
          required: false
          type: string
          default: "not_found"
        pr_url:
          description: "Pull request URL if the failure is associated with a PR in the merge queue. Omit otherwise."
          required: false
          type: string
          default: "not_found"
        author:
          description: "GitHub login of the PR author or commit author, if known. Omit otherwise."
          required: false
          type: string
          default: "not_found"
        db_entries:
          description: "Total number of unique entries currently in the CI Doctor MQ investigation database (count of distinct investigation files under /tmp/gh-aw/repo-memory/default/mq/investigations/, including the one created by this run). Report as a non-negative integer encoded as a string."
          required: true
          type: string
        occurrence_count:
          description: "How many times this same issue has been recorded in the CI Doctor MQ database, including the current investigation. Compute by matching the current failure signature (normalized error message + failure category, job-agnostic) against prior investigation/pattern files under /tmp/gh-aw/repo-memory/default/mq/. Must be >= 1. Report as a positive integer encoded as a string."
          required: true
          type: string
        statistics:
          description: "Markdown-formatted statistics summary of the CI Doctor MQ pattern database. Must include a table (or list) of every known failure pattern with: pattern signature/title, total reproduction count, first-seen timestamp (UTC, ISO 8601), and last-seen timestamp (UTC, ISO 8601). Sort patterns by reproduction count descending. Compute from files under /tmp/gh-aw/repo-memory/default/mq/investigations/ and /tmp/gh-aw/repo-memory/default/mq/patterns/. Keep concise (top 20 patterns max). Use the rendering rules from the description field (tilde fences, no raw HTML)."
          required: true
          type: string
        statistics_json:
          description: "Full statistics database serialized as a compact JSON string. Must be a JSON object of the form {\"generated_at\": <ISO8601 UTC>, \"total_patterns\": <int>, \"total_investigations\": <int>, \"patterns\": [{\"signature\": <str>, \"title\": <str>, \"category\": <str>, \"count\": <int>, \"first_seen\": <ISO8601 UTC>, \"last_seen\": <ISO8601 UTC>, \"recent_run_urls\": [<str>, ...]}]}. Include ALL known patterns, not just the top N. This payload is uploaded as a workflow artifact for offline analysis."
          required: true
          type: string
      steps:
        - name: Send Teams notification
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
            RUN_URL: ${{ github.event.workflow_run.html_url || github.event.inputs.link || '' }}
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

            ITEM=$(jq -c '[.items[] | select(.type == "notify_teams")] | last' "$GH_AW_AGENT_OUTPUT")
            if [ -z "$ITEM" ] || [ "$ITEM" = "null" ]; then
              echo "No notify_teams item present in agent output" >&2
              exit 1
            fi

            TITLE=$(echo "$ITEM"            | jq -r '.title // ""')
            FAILED_WORKFLOW=$(echo "$ITEM"  | jq -r '.failed_workflow // ""')
            PIPELINE_URL=$(echo "$ITEM"     | jq -r '.pipeline_url // ""')
            DESCRIPTION=$(echo "$ITEM"      | jq -r '.description // ""')
            PR_NUMBER=$(echo "$ITEM"        | jq -r '.pr_number // ""')
            PR_URL=$(echo "$ITEM"           | jq -r '.pr_url // ""')
            AUTHOR=$(echo "$ITEM"           | jq -r '.author // ""')
            DB_ENTRIES=$(echo "$ITEM"       | jq -r '.db_entries // ""')
            OCCURRENCES=$(echo "$ITEM"      | jq -r '.occurrence_count // ""')
            STATISTICS=$(echo "$ITEM"       | jq -r '.statistics // ""')
            STATISTICS_JSON=$(echo "$ITEM"  | jq -r '.statistics_json // ""')

            # Persist the full statistics database as a workflow artifact for offline review.
            STATS_DIR="${RUNNER_TEMP:-/tmp}/ci-doctor-mq-stats"
            mkdir -p "$STATS_DIR"
            if [ -n "$STATISTICS_JSON" ]; then
              # Validate and pretty-print; fall back to raw on parse error.
              if echo "$STATISTICS_JSON" | jq '.' > "$STATS_DIR/ci-doctor-mq-statistics.json" 2>/dev/null; then
                echo "Wrote validated statistics JSON ($(wc -c < "$STATS_DIR/ci-doctor-mq-statistics.json") bytes)"
              else
                echo "Warning: statistics_json failed jq parse; storing raw payload" >&2
                printf '%s' "$STATISTICS_JSON" > "$STATS_DIR/ci-doctor-mq-statistics.json"
              fi
            fi
            if [ -n "$STATISTICS" ]; then
              printf '%s\n' "$STATISTICS" > "$STATS_DIR/ci-doctor-mq-statistics.md"
            fi
            echo "stats_dir=$STATS_DIR" >> "$GITHUB_OUTPUT"

            # Build Adaptive Card facts conditionally (only include PR/author when present).
            FACTS=$(jq -nc \
              --arg pipeline_url    "$PIPELINE_URL" \
              --arg pr_number       "$PR_NUMBER" \
              --arg pr_url          "$PR_URL" \
              --arg author          "$AUTHOR" \
              --arg failed_workflow "$FAILED_WORKFLOW" \
              --arg db_entries      "$DB_ENTRIES" \
              --arg occurrences     "$OCCURRENCES" '
                [
                  ( $failed_workflow | select(length > 0) | { title: "Workflow",    value: . } ),
                  ( $pipeline_url    | select(length > 0) | { title: "Pipeline",    value: ("[Open run](" + . + ")") } ),
                  ( $pr_number       | select(length > 0) | { title: "PR",          value: (if ($pr_url | length) > 0 then ("[#" + . + "](" + $pr_url + ")") else ("#" + .) end) } ),
                  ( $author          | select(length > 0) | { title: "Author",      value: ("@" + .) } ),
                  ( $occurrences     | select(length > 0) | { title: "Occurrences", value: (. + "×") } ),
                  ( $db_entries      | select(length > 0) | { title: "DB entries",  value: . } )
                ] | map(select(. != null))')

            PAYLOAD=$(jq -nc \
              --arg title "$TITLE" \
              --arg description "$DESCRIPTION" \
              --arg statistics "$STATISTICS" \
              --argjson facts "$FACTS" '
                {
                  type: "message",
                  attachments: [{
                    contentType: "application/vnd.microsoft.card.adaptive",
                    content: {
                      "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                      type: "AdaptiveCard",
                      version: "1.4",
                      body: ([
                        { type: "TextBlock", text: ("\ud83d\udd34 [MQ] " + $title), weight: "Bolder", size: "Medium", color: "Attention", wrap: true },
                        { type: "FactSet", facts: $facts },
                        { type: "TextBlock", text: $description, wrap: true, spacing: "Medium" }
                      ] + (if ($statistics | length) > 0 then [
                        { type: "TextBlock", text: "Pattern Database Statistics", weight: "Bolder", size: "Medium", spacing: "Large", separator: true },
                        { type: "TextBlock", text: $statistics, wrap: true, spacing: "Small" }
                      ] else [] end))
                    }
                  }]
                }')

            curl -sS --fail-with-body \
              -H "Content-Type: application/json" \
              -d "$PAYLOAD" \
              "$TEAMS_WEBHOOK_URL"

        - name: Upload statistics artifact
          if: always()
          uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1
          with:
            name: ci-doctor-mq-statistics
            path: ${{ runner.temp }}/ci-doctor-mq-stats
            if-no-files-found: ignore
            retention-days: 90
---

# CI Doctor MQ — Teams Notification Job

Shared definition of the `notify-teams` custom safe-output job used by the
CI Doctor Merge Queue workflow. Import it via `imports:` in the consuming
workflow's frontmatter.
