---
description: |
  Shared pre-agent step for the CI Doctor Weekly Remediation workflow. Pre-downloads
  the recent CI Doctor knowledge base (investigations and per-signature failure
  patterns) from the two repo-memory branches into
  /tmp/gh-aw/agent/ci-doctor-remediation/ so the agent can start from a compact,
  pre-digested view instead of walking the memory branches itself.

  Sources (read-only):
    - memory/ci-doctor-mq          (subdirectory mq/)          — merge-queue failures
    - memory/ci-doctor-post-commit (subdirectory post-commit/) — post-commit failures

  Output layout:
    - /tmp/gh-aw/agent/ci-doctor-remediation/<slug>/patterns/<hash>.json
    - /tmp/gh-aw/agent/ci-doctor-remediation/<slug>/investigations/<timestamp>-<run-id>.json
    - /tmp/gh-aw/agent/ci-doctor-remediation/summary.txt
  where <slug> is `mq` or `post-commit`. Only records active within the last DAYS
  days (default 7) are kept.
steps:
  - name: Set up Python
    uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97  # v7.0.0
    with:
      python-version: '3.13'
  - name: Install PyGithub
    run: python -m pip install --quiet PyGithub==2.9.1
  - name: Collect CI Doctor knowledge base
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      DAYS: ${{ github.event.inputs.days || '7' }}
    run: |
      export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
      python .github/scripts/agentic-workflows/collect_ci_doctor_history.py
---

<!--
Shared CI Doctor Remediation pre-analysis step. This file has no `on:` trigger, so it
is a shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/collect-ci-doctor-history.md

Imported `steps:` are prepended to the importing workflow's own steps at compile time.
See https://github.github.com/gh-aw/reference/imports/#importing-steps
-->
