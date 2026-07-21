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
        - name: Set up Python
          uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
          with:
            python-version: '3.11'
        - name: Install PyGithub
          run: python -m pip install --quiet PyGithub
        - name: Re-add pull request to merge queue
          shell: python
          env:
            # A PAT or GitHub App token with `contents: write` and
            # `pull_requests: write` scope is required: the default GITHUB_TOKEN
            # cannot re-trigger merge_group check runs (events it creates do not
            # start new workflow runs), so the re-queued PR would stall.
            GH_TOKEN: ${{ secrets.MERGE_QUEUE_TOKEN }}
          run: |
            import json
            import os
            import re
            import subprocess
            import sys

            from github import Auth, Github

            MARKER = "<!-- ci-doctor-mq-readd -->"

            token = os.environ.get("GH_TOKEN", "")
            if not token:
                sys.exit("MERGE_QUEUE_TOKEN secret is not configured; cannot re-add to merge queue.")

            agent_output = os.environ.get("GH_AW_AGENT_OUTPUT", "")
            if not agent_output or not os.path.isfile(agent_output):
                sys.exit("No agent output found at GH_AW_AGENT_OUTPUT")

            with open(agent_output, encoding="utf-8") as handle:
                payload_items = json.load(handle).get("items", [])

            items = [it for it in payload_items if it.get("type") == "readd_to_merge_queue"]
            if not items:
                sys.exit("No readd_to_merge_queue item present in agent output")
            item = items[-1]

            pr_number = item.get("pr_number") or ""
            repository = item.get("repository") or ""
            reason = item.get("reason") or ""

            # Validate pr_number is purely numeric to avoid API path/query injection.
            if not re.fullmatch(r"[0-9]+", pr_number):
                sys.exit(f"pr_number must be a numeric string, got: '{pr_number}'")

            # Fall back to the current repository when none was supplied.
            if not repository or repository == "not_found":
                repository = os.environ.get("GITHUB_REPOSITORY", "")

            # Validate repository is in owner/repo form to avoid API path injection.
            if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository):
                sys.exit(f"repository must be in owner/repo format, got: '{repository}'")

            print(f"Requested re-add of PR #{pr_number} in {repository} to merge queue (reason: {reason})")

            github = Github(auth=Auth.Token(token))
            pull = github.get_repo(repository).get_pull(int(pr_number))

            # Loop guard: if CI Doctor already re-added this PR (marker comment
            # present), do not re-add again to avoid queue thrash.
            if any(MARKER in (comment.body or "") for comment in pull.get_issue_comments()):
                print(f"PR #{pr_number} already re-added by CI Doctor (marker comment found); skipping.")
                sys.exit(0)

            # Only re-add an open, non-draft, unmerged PR.
            if pull.merged:
                print(f"PR #{pr_number} is already merged; nothing to re-add.")
                sys.exit(0)
            if pull.state != "open":
                print(f"PR #{pr_number} is not open (state={pull.state}); not re-adding.")
                sys.exit(0)
            if pull.draft:
                print(f"PR #{pr_number} is a draft; not re-adding.")
                sys.exit(0)

            # Re-add the PR to the merge queue. On a branch that requires a merge
            # queue, `gh pr merge` adds the PR to the queue (using the queue's own
            # configured merge method) instead of merging directly. A merge-method
            # flag (`--squash`) is required so the CLI runs non-interactively.
            # `--match-head-commit` guards against the head moving under us.
            merge_command = ["gh", "pr", "merge", pr_number, "--repo", repository, "--squash"]
            if pull.head.sha:
                merge_command += ["--match-head-commit", pull.head.sha]
            merge_result = subprocess.run(merge_command, capture_output=True, text=True)
            if merge_result.returncode != 0:
                sys.exit(f"Merge-queue enqueue failed: {merge_result.stderr.strip()}")

            print(f"Successfully requested re-add of PR #{pr_number} to the merge queue.")

            # Record a marker comment so subsequent CI Doctor runs do not re-add again.
            comment_body = (
                f"{MARKER}\n\n_CI Doctor re-added this pull request to the merge queue "
                f"after a transient failure (reason: {reason})._"
            )
            pull.create_issue_comment(comment_body)

            print(f"Recorded re-add marker comment on PR #{pr_number}.")
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
