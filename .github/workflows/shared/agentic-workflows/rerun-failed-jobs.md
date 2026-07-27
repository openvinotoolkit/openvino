---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Re-runs ONLY the failed jobs of an analysed GitHub Actions workflow run,
  when the CI Doctor concludes a plain restart is likely to remedy the failure.
safe-outputs:
  jobs:
    rerun-failed-jobs:
      description: "Re-run ONLY the failed jobs of the analysed workflow run. Call this at most once and ONLY when the investigation concludes the failure is transient and a plain restart is likely to fix it (e.g. Infrastructure, Flaky Test, Network, or External Service categories such as runner hiccups, network timeouts, or downstream outages). Do NOT call for deterministic Code Issue / Dependencies / Configuration failures that a restart cannot fix."
      runs-on: ubuntu-latest
      output: "Failed jobs re-run requested for the analysed workflow run."
      permissions:
        actions: write
      inputs:
        run_id:
          description: "Numeric ID of the GitHub Actions workflow run whose failed jobs should be re-run. This is the run that was investigated (github.event.workflow_run.id for merge-queue triggers, or the run_id input for workflow_dispatch). Report as a numeric string."
          required: true
          type: string
        repository:
          description: "The owner/repo of the repository that owns the analysed run (e.g. 'openvinotoolkit/openvino'). Defaults to the current repository when omitted."
          required: false
          type: string
          default: "not_found"
        reason:
          description: "Short justification for why a restart is expected to remedy the failure (e.g. 'network timeout while downloading dependencies')."
          required: true
          type: string
      steps:
        - name: Re-run failed jobs
          env:
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          run: |
            set -euo pipefail

            if [ ! -f "${GH_AW_AGENT_OUTPUT:-}" ]; then
              echo "No agent output found at GH_AW_AGENT_OUTPUT" >&2
              exit 1
            fi

            ITEM=$(jq -c '[.items[] | select(.type == "rerun_failed_jobs")] | last' "$GH_AW_AGENT_OUTPUT")
            if [ -z "$ITEM" ] || [ "$ITEM" = "null" ]; then
              echo "No rerun_failed_jobs item present in agent output" >&2
              exit 1
            fi

            RUN_ID=$(echo "$ITEM"     | jq -r '.run_id // ""')
            REPOSITORY=$(echo "$ITEM" | jq -r '.repository // ""')
            REASON=$(echo "$ITEM"     | jq -r '.reason // ""')

            # Validate run_id is purely numeric to avoid API path injection.
            if ! printf '%s' "$RUN_ID" | grep -Eq '^[0-9]+$'; then
              echo "run_id must be a numeric string, got: '$RUN_ID'" >&2
              exit 1
            fi

            # Fall back to the current repository when none was supplied.
            if [ -z "$REPOSITORY" ] || [ "$REPOSITORY" = "not_found" ]; then
              REPOSITORY="${GITHUB_REPOSITORY}"
            fi

            # Validate repository is in owner/repo form to avoid API path injection.
            if ! printf '%s' "$REPOSITORY" | grep -Eq '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$'; then
              echo "repository must be in owner/repo format, got: '$REPOSITORY'" >&2
              exit 1
            fi

            echo "Requested re-run of failed jobs for $REPOSITORY run $RUN_ID (reason: $REASON)"

            # Guard against restart loops: if the run has already been attempted
            # more than once, do not re-run it again (mirrors rerunner.py).
            ATTEMPT=$(gh api "repos/${REPOSITORY}/actions/runs/${RUN_ID}" --jq '.run_attempt')
            if [ "${ATTEMPT:-1}" -gt 1 ]; then
              echo "Run $RUN_ID already has $ATTEMPT attempts; not re-running to avoid loops."
              exit 0
            fi

            # Re-run ONLY the failed jobs of the analysed run.
            gh api \
              --method POST \
              -H "Accept: application/vnd.github+json" \
              "repos/${REPOSITORY}/actions/runs/${RUN_ID}/rerun-failed-jobs"

            echo "Successfully requested re-run of failed jobs for $REPOSITORY run $RUN_ID."
---

# CI Doctor MQ — Re-run Failed Jobs

Shared definition of the `rerun-failed-jobs` custom safe-output job used by the
CI Doctor Merge Queue workflow. It calls the GitHub Actions
`POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs` endpoint to
restart only the failed jobs of the analysed run. Import it via `imports:` in the
consuming workflow's frontmatter, and instruct the agent to call
`rerun_failed_jobs` only when a restart is likely to remedy the failure.
