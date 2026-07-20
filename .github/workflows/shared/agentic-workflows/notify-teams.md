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
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Send Teams notification
          shell: python
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
            RUN_URL: ${{ github.event.workflow_run.html_url || github.event.inputs.link || '' }}
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
