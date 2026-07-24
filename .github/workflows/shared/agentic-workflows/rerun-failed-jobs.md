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
        contents: read
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
        - name: Checkout agentic-workflow scripts
          uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0  # v7.0.0
          with:
            sparse-checkout: .github/scripts/agentic-workflows
            persist-credentials: false
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Install PyGithub
          run: python -m pip install --quiet PyGithub
        - name: Re-run failed jobs
          env:
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          run: python .github/scripts/agentic-workflows/rerun_failed_jobs.py
---

# CI Doctor MQ — Re-run Failed Jobs

Shared definition of the `rerun-failed-jobs` custom safe-output job used by the
CI Doctor Merge Queue workflow. It calls the GitHub Actions
`POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs` endpoint to
restart only the failed jobs of the analysed run. Import it via `imports:` in the
consuming workflow's frontmatter, and instruct the agent to call
`rerun_failed_jobs` only when a restart is likely to remedy the failure.
