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
        - name: Checkout agentic-workflow scripts
          uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0  # v7.0.0
          with:
            sparse-checkout: .github/scripts/agentic-workflows
            persist-credentials: false
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Send recurring failure escalation to Teams
          env:
            TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
          run: python .github/scripts/agentic-workflows/notify_teams_recurring.py
---

# CI Doctor MQ — Recurring Teams Notification Job

Shared definition of the `notify-teams-recurring` custom safe-output job used by
the CI Doctor Merge Queue workflow. Import it via `imports:` in the consuming
workflow's frontmatter.
