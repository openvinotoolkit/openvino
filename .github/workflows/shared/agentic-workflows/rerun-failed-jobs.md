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
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Install PyGithub
          run: python -m pip install --quiet PyGithub
        - name: Re-run failed jobs
          shell: python
          env:
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          run: |
            import json
            import os
            import re
            import sys

            from github import Auth, Github

            agent_output = os.environ.get("GH_AW_AGENT_OUTPUT", "")
            if not agent_output or not os.path.isfile(agent_output):
                sys.exit("No agent output found at GH_AW_AGENT_OUTPUT")

            with open(agent_output, encoding="utf-8") as handle:
                payload_items = json.load(handle).get("items", [])

            items = [it for it in payload_items if it.get("type") == "rerun_failed_jobs"]
            if not items:
                sys.exit("No rerun_failed_jobs item present in agent output")
            item = items[-1]

            run_id = item.get("run_id") or ""
            repository = item.get("repository") or ""
            reason = item.get("reason") or ""

            # Validate run_id is purely numeric to avoid API path injection.
            if not re.fullmatch(r"[0-9]+", run_id):
                sys.exit(f"run_id must be a numeric string, got: '{run_id}'")

            # Fall back to the current repository when none was supplied.
            if not repository or repository == "not_found":
                repository = os.environ.get("GITHUB_REPOSITORY", "")

            # Validate repository is in owner/repo form to avoid API path injection.
            if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository):
                sys.exit(f"repository must be in owner/repo format, got: '{repository}'")

            print(f"Requested re-run of failed jobs for {repository} run {run_id} (reason: {reason})")

            token = os.environ.get("GH_TOKEN", "")
            if not token:
                sys.exit("GITHUB_TOKEN is not configured; cannot re-run failed jobs.")

            github = Github(auth=Auth.Token(token))
            run = github.get_repo(repository).get_workflow_run(int(run_id))

            # Guard against restart loops: if the run has already been attempted
            # more than once, do not re-run it again (mirrors rerunner.py).
            attempt = run.run_attempt or 1
            if attempt > 1:
                print(f"Run {run_id} already has {attempt} attempts; not re-running to avoid loops.")
                sys.exit(0)

            # Re-run ONLY the failed jobs of the analysed run.
            if not run.rerun_failed_jobs():
                sys.exit(f"GitHub API rejected the re-run request for {repository} run {run_id}.")

            print(f"Successfully requested re-run of failed jobs for {repository} run {run_id}.")
---

# CI Doctor MQ — Re-run Failed Jobs

Shared definition of the `rerun-failed-jobs` custom safe-output job used by the
CI Doctor Merge Queue workflow. It calls the GitHub Actions
`POST /repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs` endpoint to
restart only the failed jobs of the analysed run. Import it via `imports:` in the
consuming workflow's frontmatter, and instruct the agent to call
`rerun_failed_jobs` only when a restart is likely to remedy the failure.
