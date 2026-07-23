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
        - name: Re-add pull request to merge queue
          env:
            # A PAT or GitHub App token with `contents: write` and
            # `pull_requests: write` scope is required: the default GITHUB_TOKEN
            # cannot re-trigger merge_group check runs (events it creates do not
            # start new workflow runs), so the re-queued PR would stall.
            GH_TOKEN: ${{ secrets.MERGE_QUEUE_TOKEN }}
          run: |
            set -euo pipefail
            set +H  # disable bash history expansion so literal "!" in MARKER/comments is safe to paste/test interactively

            MARKER="<!-- ci-doctor-mq-readd -->"

            if [ -z "${GH_TOKEN:-}" ]; then
              echo "MERGE_QUEUE_TOKEN secret is not configured; cannot re-add to merge queue." >&2
              exit 1
            fi

            if [ ! -f "${GH_AW_AGENT_OUTPUT:-}" ]; then
              echo "No agent output found at GH_AW_AGENT_OUTPUT" >&2
              exit 1
            fi

            ITEM=$(jq -c '[.items[] | select(.type == "readd_to_merge_queue")] | last' "$GH_AW_AGENT_OUTPUT")
            if [ -z "$ITEM" ] || [ "$ITEM" = "null" ]; then
              echo "No readd_to_merge_queue item present in agent output" >&2
              exit 1
            fi

            PR_NUMBER=$(echo "$ITEM"  | jq -r '.pr_number // ""')
            REPOSITORY=$(echo "$ITEM" | jq -r '.repository // ""')
            REASON=$(echo "$ITEM"     | jq -r '.reason // ""')

            # Validate pr_number is purely numeric to avoid API path/query injection.
            if ! printf '%s' "$PR_NUMBER" | grep -Eq '^[0-9]+$'; then
              echo "pr_number must be a numeric string, got: '$PR_NUMBER'" >&2
              exit 1
            fi

            # Fall back to the current repository when none was supplied.
            if [ -z "$REPOSITORY" ] || [ "$REPOSITORY" = "not_found" ]; then
              REPOSITORY="${GITHUB_REPOSITORY}"
            fi

            # Validate repository is in owner/repo form to avoid API path injection.
            if ! printf '%s' "$REPOSITORY" | grep -Eq '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$'; then
              echo "repository must be in owner/repo format, got: '$REPOSITORY'" >&2
              exit 1
            fi

            echo "Requested re-add of PR #$PR_NUMBER in $REPOSITORY to merge queue (reason: $REASON)"

            # Loop guard: if CI Doctor already re-added this PR (marker comment
            # present), do not re-add again to avoid queue thrash.
            if gh pr view "$PR_NUMBER" --repo "$REPOSITORY" --json comments \
                 --jq '.comments[].body' | grep -qF "$MARKER"; then
              echo "PR #$PR_NUMBER already re-added by CI Doctor (marker comment found); skipping."
              exit 0
            fi

            # Fetch PR state. Note: `gh pr view --json` does not expose merge-queue
            # membership, so this job does not re-check whether the PR is still in
            # the queue — that decision is made upstream by the agent (which only
            # calls this tool when the PR was dropped), and re-running `gh pr merge`
            # on an already-queued PR is safe.
            PR_INFO=$(gh pr view "$PR_NUMBER" --repo "$REPOSITORY" \
              --json state,isDraft,headRefOid)

            PR_STATE=$(echo "$PR_INFO" | jq -r '.state // ""')
            PR_DRAFT=$(echo "$PR_INFO" | jq -r '.isDraft // false')
            PR_HEAD=$(echo "$PR_INFO"  | jq -r '.headRefOid // ""')

            # Only re-add an open, non-draft, unmerged PR.
            if [ "$PR_STATE" = "MERGED" ]; then
              echo "PR #$PR_NUMBER is already merged; nothing to re-add."
              exit 0
            fi
            if [ "$PR_STATE" != "OPEN" ]; then
              echo "PR #$PR_NUMBER is not open (state=$PR_STATE); not re-adding."
              exit 0
            fi
            if [ "$PR_DRAFT" = "true" ]; then
              echo "PR #$PR_NUMBER is a draft; not re-adding."
              exit 0
            fi

            # Re-add the PR to the merge queue. On a branch that requires a merge
            # queue, `gh pr merge` adds the PR to the queue (or enables auto-merge
            # if required checks are still pending) instead of merging directly,
            # using the queue's own configured merge method. A merge-method flag
            # (`--squash`) is still required so the CLI runs non-interactively
            # (otherwise it prompts / errors asking which method to use).
            # `--match-head-commit` guards against the head moving under us.
            if [ -n "$PR_HEAD" ]; then
              gh pr merge "$PR_NUMBER" --repo "$REPOSITORY" --squash --match-head-commit "$PR_HEAD"
            else
              gh pr merge "$PR_NUMBER" --repo "$REPOSITORY" --squash
            fi

            echo "Successfully requested re-add of PR #$PR_NUMBER to the merge queue."

            # Record a marker comment so subsequent CI Doctor runs do not re-add again.
            COMMENT_BODY=$(printf '%s\n\n_CI Doctor re-added this pull request to the merge queue after a transient failure (reason: %s)._' \
              "$MARKER" "$REASON")
            gh pr comment "$PR_NUMBER" --repo "$REPOSITORY" --body "$COMMENT_BODY"

            echo "Recorded re-add marker comment on PR #$PR_NUMBER."
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
