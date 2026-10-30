---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Proposes a new workflow-rerunner search string for HUMAN REVIEW. The agent
  never writes rerunner-consumed strings itself: this job validates the
  candidate deterministically (transient category, specificity, verbatim log
  evidence, no overlap with errors_to_look_for.json or open proposals, no
  previously proposed signature) and, if it passes, opens a pull request against
  the default branch appending one entry to
  .github/scripts/workflow_rerun/errors_to_look_for.json. The string becomes
  active only once a maintainer merges the PR.
safe-outputs:
  jobs:
    propose-rerun-pattern:
      description: "Propose a workflow-rerunner search string for human review. Call at most once, and ONLY for a transient failure (Flaky Test, Infrastructure, Network, External Service) whose verified rerun_search_string candidate is non-null and not already approved. The job validates the candidate and opens a pull request adding it to .github/scripts/workflow_rerun/errors_to_look_for.json; nothing is re-run until a maintainer merges it. Rejected or duplicate candidates are skipped silently."
      runs-on: ubuntu-latest
      output: "Rerun pattern proposal submitted for review."
      permissions:
        contents: read         # sparse-checkout of the job script; all writes use RERUNNER_DB_UPDATE
      inputs:
        signature_hash:
          description: "The <signature-hash> of the current failure's pattern record (file name of mq/patterns/<signature-hash>.json)."
          required: true
          type: string
        rerun_search_string:
          description: "The verified rerun_search_string candidate: a short, stable substring copied from a single raw failure log line, specific to this transient failure (not a step name or generic phrase)."
          required: true
          type: string
        category:
          description: "Failure category; must be one of Flaky Test, Infrastructure, Network, External Service."
          required: true
          type: string
        title:
          description: "Short human-readable failure title (same as notify_teams.title)."
          required: true
          type: string
        sample_log_line:
          description: "The verbatim raw log line (from the pre-downloaded logs) that contains rerun_search_string. Used as review evidence and re-verified by the job."
          required: true
          type: string
        pipeline_url:
          description: "URL of the analysed failed workflow run (evidence run)."
          required: true
          type: string
        reason:
          description: "One-line justification of why a plain rerun fixes this failure."
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
        - name: Propose rerun pattern
          env:
            GH_TOKEN: ${{ secrets.RERUNNER_DB_UPDATE }}
            ERRORS_TO_LOOK_FOR_PATH: .github/scripts/workflow_rerun/errors_to_look_for.json
            PR_LABELS: "category: CI,github_actions,agentic-workflows"
            PR_REVIEWERS: akashchi
          run: |
            export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
            python .github/scripts/agentic-workflows/propose_rerun_pattern.py
---

# CI Doctor MQ — Propose Rerun Pattern

Shared definition of the `propose-rerun-pattern` custom safe-output job used by
the CI Doctor Merge Queue workflow.

The workflow rerunner (`.github/workflows/workflow_rerunner.yml`) re-runs every
failed monitored pipeline whose log contains a search string from
`.github/scripts/workflow_rerun/errors_to_look_for.json`. A string that is too
generic makes it restart almost every failing pipeline, so new strings are never
added by the agent directly. Instead:

1. The agent computes and verifies a `rerun_search_string` candidate (stored
   only as informational metadata on its `mq/patterns/<hash>.json` record) and
   calls `propose_rerun_pattern`.
2. This job (`.github/scripts/agentic-workflows/propose_rerun_pattern.py`)
   re-validates it:
   - transient category only;
   - at least 3 words / 15 characters after normalization, not made up only of
     generic words or runner boilerplate, not a hash, no URL;
   - a substring of the supplied verbatim `sample_log_line`, which must not be a
     `##[group]` step header;
   - no overlap with `errors_to_look_for.json` on the default branch or with
     entries added by other open proposal PRs;
   - the signature has never been proposed before (a closed PR counts as a
     rejection and is not re-proposed).
3. If it passes, the job creates branch `ci-doctor/rerun-pattern/<hash>` off the
   default branch, appends `{"error_text", "ticket": 9999999, "category"}` to
   `errors_to_look_for.json`, and opens a PR with the configured labels and
   reviewers.

Merging the PR activates the string; closing it rejects it permanently. The
branch, commit and PR are created with the `RERUNNER_DB_UPDATE` token (not the
workflow `GITHUB_TOKEN`), so the PR triggers CI like any other PR.
