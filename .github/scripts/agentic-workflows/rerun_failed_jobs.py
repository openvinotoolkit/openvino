#!/usr/bin/env python3
"""Re-run ONLY the failed jobs of an analysed GitHub Actions workflow run.

Used by the `rerun-failed-jobs` custom safe-output job of the CI Doctor MQ
workflow (.github/workflows/shared/agentic-workflows/rerun-failed-jobs.md).
Reads the agent output referenced by GH_AW_AGENT_OUTPUT and calls the GitHub
Actions rerun-failed-jobs endpoint via PyGithub.
"""

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
