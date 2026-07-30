---
description: |
  Shared pre-agent step for the CI Doctor merge-queue workflow. Resolves the
  pull request behind a merge-queue (`merge_group`) run and pre-collects its
  metadata into /tmp/gh-aw/agent/ci-doctor/.

  Output layout:
    - /tmp/gh-aw/agent/ci-doctor/pr-info.json  (structured PR metadata)
    - /tmp/gh-aw/agent/ci-doctor/pr-info.txt   (human-readable summary)
steps:
  - name: Set up Python
    uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
    with:
      python-version: '3.13'
  - name: Install PyGithub
    run: python -m pip install --quiet PyGithub
  - name: Collect PR info
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
    run: |
      export PYTHONPATH=.github/scripts/agentic-workflows/:${PYTHONPATH}
      python .github/scripts/agentic-workflows/collect_pr_info.py
---

<!--
Shared CI Doctor pre-analysis step. This file has no `on:` trigger, so it is a
shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/collect-pr-info.md

Imported `steps:` are prepended to the importing workflow's own steps at compile
time. See https://github.github.com/gh-aw/reference/imports/#importing-steps
-->
