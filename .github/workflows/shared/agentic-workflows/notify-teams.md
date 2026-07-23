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
        - name: Checkout agentic-workflow scripts
          uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0  # v7.0.0
          with:
            sparse-checkout: .github/scripts/agentic-workflows
            persist-credentials: false
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Send Teams notification
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
            RUN_URL: ${{ github.event.workflow_run.html_url || github.event.inputs.link || '' }}
          run: python .github/scripts/agentic-workflows/notify_teams.py

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
