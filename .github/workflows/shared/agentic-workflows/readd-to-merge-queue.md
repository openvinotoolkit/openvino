---
description: |
  Shared custom safe-output job for the CI Doctor MQ workflow.
  Re-adds a pull request that was dropped from the GitHub merge queue as a result
  of a transient pipeline failure back into the merge queue via the `gh pr merge`
  command.
safe-outputs:
  jobs:
    readd-to-merge-queue:
      description: "Re-add a pull request that was removed from the merge queue because of the analysed pipeline failure back into the merge queue. Call this at most once and ONLY when the investigation concludes the failure is transient and a plain re-queue is likely to let the PR merge (e.g. Infrastructure, Flaky Test, Network, or External Service categories such as runner hiccups, network timeouts, cancelled jobs, or downstream outages). Do NOT call for deterministic Code Issue / Dependencies / Configuration failures that re-queuing cannot fix."
      runs-on: ubuntu-latest
      output: "Re-add to merge queue requested for the affected pull request."
      permissions:
        pull-requests: read
        contents: read
      inputs:
        pr_number:
          description: "Number of the pull request that was dropped from the merge queue and should be re-added. Report as a numeric string."
          required: true
          type: string
        repository:
          description: "The owner/repo of the repository that owns the pull request (e.g. 'openvinotoolkit/openvino'). Defaults to the current repository when omitted."
          required: false
          type: string
          default: "not_found"
        reason:
          description: "Short justification for why re-queuing is expected to let the PR merge (e.g. 'network timeout while downloading dependencies')."
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
        - name: Re-add pull request to merge queue
          env:
            # A PAT or GitHub App token with `contents: write` and
            # `pull_requests: write` scope is required: the default GITHUB_TOKEN
            # cannot re-trigger merge_group check runs (events it creates do not
            # start new workflow runs), so the re-queued PR would stall.
            GH_TOKEN: ${{ secrets.MERGE_QUEUE_TOKEN }}
          run: python .github/scripts/agentic-workflows/readd_to_merge_queue.py
---

# CI Doctor MQ — Re-add Pull Request to Merge Queue

Shared definition of the `readd-to-merge-queue` custom safe-output job used by
the CI Doctor Merge Queue workflow. When a merge-queue pipeline fails, GitHub
drops the affected pull request from the merge queue. If the CI Doctor concludes
the failure was transient, this job re-adds the PR to the queue via the
`gh pr merge` command (on a branch that requires a merge queue, `gh pr merge`
enqueues the PR instead of merging it directly, using the queue's own configured
merge method). A `--squash` flag is passed so the command runs non-interactively;
on a merge-queue branch the server ignores it in favour of the queue's method.

It requires a `MERGE_QUEUE_TOKEN` secret holding a PAT or GitHub App token with
`contents: write` and `pull_requests: write` scope — the default `GITHUB_TOKEN`
cannot re-trigger `merge_group` check runs, so a re-queue performed with it would
stall.

The job is idempotent and loop-safe: it skips the PR when it is already merged,
closed, or a draft, or when a previous CI Doctor re-add marker comment is
present. Import it via `imports:` in the consuming workflow's frontmatter, and
instruct the agent to call `readd_to_merge_queue` only when a re-queue is likely
to let the PR merge.
