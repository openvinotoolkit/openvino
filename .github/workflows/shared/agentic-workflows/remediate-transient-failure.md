---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Remediates a *transient* merge-queue pipeline failure. The agent calls it
  whenever the failure is transient and knows nothing about the PR's merge-queue
  status; this job resolves that status live and decides on its own whether to
  re-run only the failed jobs (PR still queued) or re-add the PR to the merge
  queue (PR was dropped on the failure).
safe-outputs:
  jobs:
    remediate-transient-failure:
      description: "Remediate a TRANSIENT merge-queue pipeline failure. Call this at most once and ONLY when the investigation concludes the failure is transient (Infrastructure, Flaky Test, Network, or External Service categories such as runner hiccups, network timeouts, cancelled jobs, or downstream outages). The job resolves the PR's live merge-queue status itself and decides whether to re-run the failed jobs (PR still in the queue) or re-add the PR to the merge queue (PR was dropped) — you do NOT know or provide the queue status. Do NOT call for deterministic Code Issue / Dependencies / Configuration failures that a restart or re-queue cannot fix."
      runs-on: ubuntu-latest
      output: "Transient-failure remediation requested for the analysed workflow run."
      permissions:
        actions: write         # re-run the failed jobs
        contents: read
        pull-requests: write   # read live merge-queue status; post the re-run outcome comment
      inputs:
        run_id:
          description: "Numeric ID of the analysed GitHub Actions workflow run (github.event.workflow_run.id for merge-queue triggers, or the run_id input for workflow_dispatch). Report as a numeric string."
          required: true
          type: string
        repository:
          description: "The owner/repo of the repository that owns the analysed run (e.g. 'openvinotoolkit/openvino'). Defaults to the current repository when omitted."
          required: false
          type: string
          default: "not_found"
        reason:
          description: "Short justification for why the failure is transient and remediation is expected to help (e.g. 'network timeout while downloading dependencies')."
          required: true
          type: string
      steps:
        - name: Checkout agentic-workflow scripts
          uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1
          with:
            sparse-checkout: .github/scripts/agentic-workflows
            persist-credentials: false
        - name: Set up Python
          uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97  # v7.0.0
          with:
            python-version: '3.13'
        - name: Install PyGithub
          run: python -m pip install --quiet PyGithub==2.9.1
        - name: Remediate transient failure
          env:
            # GITHUB_TOKEN (with actions: write) re-runs the failed jobs and reads
            # the live merge-queue status. MERGE_QUEUE_TOKEN (a PAT or GitHub App
            # token with contents: write + pull_requests: write) is required to
            # re-add a dropped PR: the default GITHUB_TOKEN cannot re-trigger
            # merge_group check runs, so a re-queue performed with it would stall.
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            MERGE_QUEUE_TOKEN: ${{ secrets.MERGE_QUEUE_TOKEN }}
          run: |
            export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
            python .github/scripts/agentic-workflows/remediate_transient_failure.py
        # The `echo` steps are for collecting and aggregating statistics in Grafana
        - name: Re-run of failed jobs was triggered
          if: ${{ env.RERUN_FAILED_JOBS == 'true' }}
          run: echo "CI Doctor re-ran the failed jobs for the analysed run (PR still in the merge queue)."
        - name: Re-add to merge queue was triggered
          if: ${{ env.READDED_TO_MQ == 'true' }}
          run: echo "CI Doctor re-added the pull request to the merge queue (it had been dropped)."
---

# CI Doctor MQ — Remediate Transient Failure

Shared definition of the `remediate-transient-failure` custom safe-output job
used by the CI Doctor Merge Queue workflow. It merges the former
`rerun-failed-jobs` and `readd-to-merge-queue` jobs into a single remediation
entry point so the agent only has to decide *whether the failure is transient*,
not *which* remedy applies.

When invoked, the job:

1. Resolves the PR behind the analysed run (from the
   `gh-readonly-queue/<base>/pr-<number>-<sha>` head branch, falling back to the
   run's associated pull requests).
2. Reads the PR's **live** merge-queue status (`common.merge_queue_status`, which
   prefers the GraphQL `mergeQueue` entries and falls back to the PR's add/remove
   timeline events). Doing this at action time avoids acting on a stale snapshot.
3. Picks the remedy:
   - **`in_queue`** — re-runs only the failed jobs via
     `POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs`
     (refuses a run already on its second attempt, to avoid restart loops) and
     posts an outcome comment on the PR.
   - **`not_in_queue`** — re-adds the PR via `gh pr merge` (which enqueues on a
     merge-queue branch) and posts an outcome comment carrying the re-add marker.
     Idempotent and loop-safe: it skips the PR when it is already merged, closed,
     a draft, or when a previous CI Doctor re-add marker comment is present.
   - **`merged`** / **`unknown`** — takes no action (fail safe).

It uses two tokens: the default `GITHUB_TOKEN` (with `actions: write`) re-runs the
failed jobs and reads queue status, and a `MERGE_QUEUE_TOKEN` secret (a PAT or
GitHub App token with `contents: write` + `pull_requests: write`) performs the
re-queue — the default `GITHUB_TOKEN` cannot re-trigger `merge_group` check runs,
so a re-queue performed with it would stall.

Import it via `imports:` in the consuming workflow's frontmatter, and instruct
the agent to call `remediate_transient_failure` only when the failure is
transient.
