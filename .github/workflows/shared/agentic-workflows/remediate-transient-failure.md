---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Handles a *transient* merge-queue pipeline failure by re-running failed jobs
  if the PR is still queued and posting a trusted eligibility marker. A
  separate deterministic workflow validates the marker and dequeue reason
  before it re-adds the PR.
safe-outputs:
  jobs:
    remediate-transient-failure:
      description: "Mark a TRANSIENT merge-queue pipeline failure as eligible for re-queueing. Call this at most once and ONLY when the investigation concludes the failure is transient (Infrastructure, Flaky Test, Network, or External Service categories such as runner hiccups, network timeouts, cancelled jobs, or downstream outages). The job resolves the PR's live queue status, re-runs failed jobs if the PR is still queued, and posts a bot-authored marker bound to the current PR head and failed run. It never re-adds the PR; the separate dequeue workflow validates the marker and reason before re-queueing. Do NOT call for deterministic Code Issue / Dependencies / Configuration failures."
      runs-on: ubuntu-latest
      output: "Transient-failure eligibility marker requested for the analysed workflow run."
      permissions:
        actions: write         # re-run the failed jobs
        contents: read
        pull-requests: write   # read live merge-queue status and post the eligibility marker
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
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          run: |
            export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
            python .github/scripts/agentic-workflows/remediate_transient_failure.py
        # The `echo` steps are for collecting and aggregating statistics in Grafana.
        - name: Re-run of failed jobs was triggered
          if: ${{ env.RERUN_FAILED_JOBS == 'true' }}
          run: echo "CI Doctor re-ran the failed jobs for the analysed run (PR still in the merge queue)."
        - name: Transient eligibility marker was posted
          if: ${{ env.TRANSIENT_ELIGIBILITY_MARKED == 'true' }}
          run: echo "CI Doctor marked the PR as eligible for re-queueing after a CI_FAILURE dequeue."
---

# CI Doctor MQ — Remediate Transient Failure

Shared definition of the `remediate-transient-failure` custom safe-output job
used by the CI Doctor Merge Queue workflow. The agent decides only whether the
failure is transient; this job handles a safe re-run when the PR remains queued
and emits the trusted marker consumed by the deterministic re-queue workflow.

When invoked, the job:

1. Resolves the PR behind the analysed run (from the
   `gh-readonly-queue/<base>/pr-<number>-<sha>` head branch, falling back to the
   run's associated pull requests).
2. Reads the PR's **live** merge-queue status (`common.merge_queue_status`, which
   prefers the GraphQL `mergeQueue` entries and falls back to the PR's add/remove
   timeline events).
3. If the PR is still queued, re-runs only the failed jobs via
   `POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs`. It
   refuses a run already on its second attempt to avoid restart loops.
4. Posts a bot-authored comment with a visible eligibility explanation and a
   machine-readable marker bound to the PR number, current head SHA, failed run
   ID, and analysis run ID. It does this whether the PR is still queued or has
   already been dequeued.
5. Does not re-add the PR. The separate workflow validates the marker, the
   latest `CI_FAILURE` dequeue event, current PR head, and live queue status before
   using `MERGE_QUEUE_TOKEN` to re-queue it.

Re-running uses the default ``GITHUB_TOKEN`` with ``actions: write``; the
separate re-queue workflow uses ``MERGE_QUEUE_TOKEN`` because the default token
cannot re-trigger ``merge_group`` check runs.

Import it via `imports:` in the consuming workflow's frontmatter, and instruct
the agent to call `remediate_transient_failure` only when the failure is
transient.
