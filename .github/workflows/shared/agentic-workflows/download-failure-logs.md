---
description: |
  Shared pre-agent step for the CI Doctor workflows. Pre-downloads failed CI logs
  and pre-locates error hints into /tmp/gh-aw/agent/ci-doctor/ so the agent can
  start from a compact summary instead of re-downloading logs itself.

  The step auto-detects its mode from the environment (no parameters required):
    - run mode  (RUN_ID set):    analyse a single workflow run (CI Doctor — Merge Queue).
    - pr mode   (PR_NUMBER set):  analyse every failed run on a pull request head commit.

  Output layout (identical in both modes):
    - /tmp/gh-aw/agent/ci-doctor/logs/job-<job-id>.log
    - /tmp/gh-aw/agent/ci-doctor/filtered/job-<job-id>-hints.txt
    - /tmp/gh-aw/agent/ci-doctor/summary.txt
  Run mode additionally writes logs/failed-jobs.json; PR mode additionally writes
  logs/failed-runs.json and logs/run-<run-id>-failed-jobs.json.
steps:
  - name: Set up Python
    uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
    with:
      python-version: '3.11'
  - name: Install PyGithub
    run: python -m pip install --quiet PyGithub
  - name: Download CI failure logs
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      PR_NUMBER: ${{ github.event.issue.number }}
      RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
    run: python .github/scripts/agentic-workflows/download_failure_logs.py
---

<!--
Shared CI Doctor pre-analysis step. This file has no `on:` trigger, so it is a
shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/download-failure-logs.md

Imported `steps:` are prepended to the importing workflow's own steps at compile
time. See https://github.github.com/gh-aw/reference/imports/#importing-steps
-->
