---
name: "CI Failure Doctor — Post-Commit"
description: |
  This workflow is an automated CI failure investigator for post-commit (push)
  runs on the OpenVINO repository. It triggers when monitored workflows fail on a
  push to a protected branch, performs a deep analysis of the failure, records the
  problem in a persistent knowledge base, and reports it via a Microsoft Teams
  notification. Unlike the Merge Queue doctor, it never re-runs or re-queues
  pipelines — it only collects and reports.

  The investigation protocol shared with the Merge Queue doctor lives in the
  parameterised `shared/agentic-workflows/ci-doctor-*.md` fragments imported below;
  the body of this file only holds the post-commit-specific additions.

on:
  workflow_dispatch:
    inputs:
      run_id:
        description: "Workflow run ID to investigate (for manual testing)"
        required: false
      link:
         description: "Link to a workflow to investigate (for manual testing across repositories)"
         required: false
  workflow_run:
    workflows:
      - "Linux (Ubuntu 22.04, Python 3.11)"
      - "Linux (Ubuntu 24.04, Python 3.12)"
      - "Android"
      - "Linux ARM64 (Ubuntu 22.04, Python 3.11)"
      - "Linux (Ubuntu 22.04, ARM64 cross-compilation, Python 3.11)"
      - "Linux Static CC (Ubuntu 22.04, Python 3.11, Clang)"
      - "Linux RISC-V (Ubuntu 22.04, Python 3.10)"
      - "Windows (VS 2022, Python 3.11, Release)"
      - "Windows (VS 2022, Python 3.11, Debug)"
      - "Windows Conditional Compilation (VS 2022, Python 3.11)"
      - "Webassembly"
      - "Manylinux 2_28"
      - "Clang-tidy static analysis (Ubuntu 24.04, Python 3.12, Clang-18, Clang-tidy-18)"
    types:
      - completed
    branches:
      - master
  # Post-commit pushes to master land via the merge queue, so the triggering actor is a bot
  bots: [github-merge-queue]
concurrency:
  group: gh-aw-${{ github.workflow }}

# Only trigger for post-commit (push) failures on the master branch (the workflow_run `branches` filter
# already restricts triggering runs to master), or manual workflow_dispatch for testing
if: ${{ github.event_name == 'workflow_dispatch' || (github.event.workflow_run.conclusion == 'failure' && github.event.workflow_run.event == 'push') }}

permissions: read-all

model: claude-sonnet-5
engine:
  id: copilot
network: defaults

imports:
  # Pre-agent steps (log pre-download)
  - shared/agentic-workflows/download-failure-logs.md
  # Safe-output jobs
  - shared/agentic-workflows/notify-teams.md
  # Shared CI Doctor prompt (inlined before this file's body, in this order).
  # Also wires tools.github, tools.repo-memory (memory/ci-doctor-post-commit) and
  # the investigations/patterns artifact upload.
  - uses: shared/agentic-workflows/ci-doctor-investigation-protocol.md
    with:
      name: Post-Commit
      kind: post-commit
      event: push
      slug: post-commit
  - uses: shared/agentic-workflows/ci-doctor-knowledge-base.md
    with:
      slug: post-commit
      index_scope_field: commit_sha
      pattern_scope_field: affected_commits
      scope_ref: "the head SHA of the current workflow run (or null if unknown)"
  - uses: shared/agentic-workflows/ci-doctor-reporting.md
    with:
      name: Post-Commit
      kind: post-commit
      event: push
      source: post_commit
      slug: post-commit

safe-outputs:
  report-failure-as-issue:   # defeat the silent "produced no safe outputs" no-op
    - agent_failure
    - missing_safe_outputs
    - timed_out

timeout-minutes: 20

---

# CI Failure Doctor — Post-Commit: Workflow-Specific Instructions

The shared CI Doctor protocol above applies in full. The sections below narrow it for the **post-commit** investigator and take precedence wherever they conflict with it.

## Scope

- Only failures of `push`-triggered runs on the `master` branch are investigated; the `workflow_run` `branches` filter in this workflow already restricts triggering runs to `master`, so no additional branch check is needed.
- **Report only**: this doctor never re-runs, restarts, re-queues or otherwise remediates any pipeline, and never comments on pull requests. It collects and reports, nothing more.

## Pull Request Context

Post-commit failures are **commit-scoped**: no PR metadata is pre-collected for this doctor.

- In Phase 3 step 4, the failing commit can sometimes be traced back to a merged PR through the run's head commit metadata (e.g. the merge-commit message or the commit's associated PR). When that mapping is clear, note the PR number/author for the report and use it as the PR context for Phase 4; otherwise treat the failure as commit-scoped.
- For the optional `notify_teams` fields `pr_number` / `pr_url` / `author`: provide them only when the failing commit clearly maps to a merged PR / commit author. Omit or pass `"not_found"` when they cannot be determined — post-commit failures frequently have no directly associated PR context.

## Mandatory Output Requirement — Post-Commit Safe Outputs

This workflow has **no** safe-output tools beyond the shared `notify_teams`, `noop` and `missing_data`. Call **exactly one** of them per run:

- `notify_teams` (with `source: "post_commit"`) — any actionable finding.
- `noop` — no investigation needed (successful run, not a post-commit run, nothing to investigate).
- `missing_data` — the investigation is blocked by missing data.
