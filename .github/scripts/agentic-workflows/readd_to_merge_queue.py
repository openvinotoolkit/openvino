#!/usr/bin/env python3
"""Re-add a pull request dropped from the merge queue back into the queue.

Used by the `readd-to-merge-queue` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/readd-to-merge-queue.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT, and re-adds the PR via
`gh pr merge` (which enqueues on a merge-queue branch). It is idempotent: it
skips merged/closed/draft PRs and PRs already carrying a re-add marker comment.
"""

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
