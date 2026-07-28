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
  - name: Download CI failure logs
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      PR_NUMBER: ${{ github.event.issue.number }}
      RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
    run: |
      set -e
      LOG_DIR="/tmp/gh-aw/agent/ci-doctor/logs"
      FILTERED_DIR="/tmp/gh-aw/agent/ci-doctor/filtered"
      SUMMARY_FILE="/tmp/gh-aw/agent/ci-doctor/summary.txt"
      mkdir -p "$LOG_DIR" "$FILTERED_DIR"

      # Download the log for a single failed job and pre-locate error hints.
      download_job_log() {
        JOB_ID="$1"
        LOG_FILE="$LOG_DIR/job-${JOB_ID}.log"
        echo "Downloading log for job $JOB_ID..."
        gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" > "$LOG_FILE" 2>/dev/null \
          || echo "(log download failed)" > "$LOG_FILE"
        echo "  -> Saved $(wc -l < "$LOG_FILE") lines to $LOG_FILE"

        HINTS_FILE="$FILTERED_DIR/job-${JOB_ID}-hints.txt"
        grep -n -m 30 -iE "(error[: ]|ERROR|FAIL|panic:|fatal[: ]|undefined[: ]|exception|exit status [^0])" \
          "$LOG_FILE" > "$HINTS_FILE" 2>/dev/null || true
        if [ -s "$HINTS_FILE" ]; then
          echo "  -> Pre-located $(wc -l < "$HINTS_FILE") hint line(s) in $HINTS_FILE"
        fi
      }

      # Append the shared "downloaded files" + "hint files" footer to the summary.
      write_summary_footer() {
        echo ""
        echo "Downloaded job log files ($LOG_DIR):"
        for LOG_FILE in "$LOG_DIR"/job-*.log; do
          [ -f "$LOG_FILE" ] || continue
          echo "  $LOG_FILE ($(wc -l < "$LOG_FILE") lines)"
        done
        echo ""
        echo "Filtered hint files ($FILTERED_DIR):"
        for HINTS_FILE in "$FILTERED_DIR"/*-hints.txt; do
          [ -s "$HINTS_FILE" ] || continue
          echo "  $HINTS_FILE ($(wc -l < "$HINTS_FILE") matches)"
          head -3 "$HINTS_FILE" | sed 's/^/    /'
        done
      }

      if [ -n "$RUN_ID" ]; then
        # ---- run mode: a single workflow run (CI Doctor — Merge Queue) ----
        echo "=== CI Doctor: Pre-downloading logs for run $RUN_ID ==="

        gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" \
          --jq '[.jobs[] | select(.conclusion == "failure" or .conclusion == "cancelled") | {id:.id, name:.name, failed_steps:[.steps[]? | select(.conclusion=="failure") | .name]}]' \
          > "$LOG_DIR/failed-jobs.json"

        FAILED_COUNT=$(jq 'length' "$LOG_DIR/failed-jobs.json")
        echo "Found $FAILED_COUNT failed job(s)"

        {
          echo "=== CI Doctor Pre-Analysis ==="
          echo "Run ID: $RUN_ID"
          echo ""
          echo "Failed jobs (details in $LOG_DIR/failed-jobs.json):"
          jq -r '.[] | "  Job \(.id): \(.name)\n    Failed steps: \(.failed_steps | join(", "))"' \
            "$LOG_DIR/failed-jobs.json"
        } > "$SUMMARY_FILE"

        if [ "$FAILED_COUNT" -eq 0 ]; then
          echo "No failed jobs found, skipping log download"
        else
          cat "$LOG_DIR/failed-jobs.json"
          jq -r '.[].id' "$LOG_DIR/failed-jobs.json" | while read -r JOB_ID; do
            download_job_log "$JOB_ID"
          done
        fi

        write_summary_footer >> "$SUMMARY_FILE"

      elif [ -n "$PR_NUMBER" ]; then
        # ---- pr mode: every failed run on a pull request head commit ----
        echo "=== CI Doctor: Pre-downloading logs for PR #$PR_NUMBER ==="

        HEAD_SHA=$(gh api "repos/$REPO/pulls/$PR_NUMBER" --jq '.head.sha' 2>/dev/null || echo "")
        if [ -z "$HEAD_SHA" ]; then
          echo "Could not resolve a pull request head SHA (is this a PR comment?), skipping log download"
          echo "No pull request context available." > "$SUMMARY_FILE"
          exit 0
        fi
        echo "PR head SHA: $HEAD_SHA"

        # Find all workflow runs for the PR head SHA that failed or were cancelled
        gh api --paginate "repos/$REPO/actions/runs?head_sha=$HEAD_SHA" \
          --jq '[.workflow_runs[] | select(.conclusion == "failure" or .conclusion == "cancelled") | {id:.id, name:.name, url:.html_url, conclusion:.conclusion}]' \
          > "$LOG_DIR/failed-runs.json" 2>/dev/null || echo '[]' > "$LOG_DIR/failed-runs.json"

        # De-duplicate by workflow name, keeping the most recent (highest id) run per workflow
        jq 'group_by(.name) | map(max_by(.id))' "$LOG_DIR/failed-runs.json" > "$LOG_DIR/failed-runs.dedup.json"
        mv "$LOG_DIR/failed-runs.dedup.json" "$LOG_DIR/failed-runs.json"

        FAILED_COUNT=$(jq 'length' "$LOG_DIR/failed-runs.json")
        echo "Found $FAILED_COUNT failed pipeline(s) on PR #$PR_NUMBER"

        {
          echo "=== CI Doctor Pre-Analysis (PR #$PR_NUMBER, head $HEAD_SHA) ==="
          echo ""
          echo "Failed pipelines (details in $LOG_DIR/failed-runs.json):"
          jq -r '.[] | "  Run \(.id): \(.name) [\(.conclusion)] \(.url)"' "$LOG_DIR/failed-runs.json"
        } > "$SUMMARY_FILE"

        if [ "$FAILED_COUNT" -eq 0 ]; then
          echo "No failed pipelines found, skipping log download"
        else
          cat "$LOG_DIR/failed-runs.json"
          jq -r '.[].id' "$LOG_DIR/failed-runs.json" | while read -r RUN_ID; do
            gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" \
              --jq '[.jobs[] | select(.conclusion == "failure" or .conclusion == "cancelled") | {id:.id, name:.name, failed_steps:[.steps[]? | select(.conclusion=="failure") | .name]}]' \
              > "$LOG_DIR/run-${RUN_ID}-failed-jobs.json" 2>/dev/null || echo '[]' > "$LOG_DIR/run-${RUN_ID}-failed-jobs.json"

            jq -r '.[].id' "$LOG_DIR/run-${RUN_ID}-failed-jobs.json" | while read -r JOB_ID; do
              download_job_log "$JOB_ID"
            done
          done
        fi

        write_summary_footer >> "$SUMMARY_FILE"

      else
        echo "Neither RUN_ID nor PR_NUMBER is set; nothing to pre-download."
        echo "No CI Doctor context (no RUN_ID or PR_NUMBER)." > "$SUMMARY_FILE"
      fi

      echo ""
      echo "✅ Pre-analysis complete. Agent should start with $SUMMARY_FILE"
---

<!--
Shared CI Doctor pre-analysis step. This file has no `on:` trigger, so it is a
shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/download-failure-logs.md

Imported `steps:` are prepended to the importing workflow's own steps at compile
time. See https://github.github.com/gh-aw/reference/imports/#importing-steps
-->
