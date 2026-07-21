---
description: |
  Shared pre-agent step for the CI Doctor workflows. Pre-downloads failed CI logs
  and pre-locates error hints into /tmp/gh-aw/agent/ci-doctor/ so the agent can
  start from a compact summary instead of re-downloading logs itself.

  The step auto-detects its mode from the environment (no parameters required):
    - run mode  (RUN_ID set):    analyse a single workflow run (CI Doctor — Merge Queue).
    - pr mode   (PR_NUMBER set):  analyse every failed run on a pull request head commit.

  Output layout (identical in both modes):
    - /tmp/gh-aw/agent/ci-doctor/logs/job-<job-id>.log
    - /tmp/gh-aw/agent/ci-doctor/filtered/job-<job-id>-hints.txt
    - /tmp/gh-aw/agent/ci-doctor/summary.txt
  Run mode additionally writes logs/failed-jobs.json; PR mode additionally writes
  logs/failed-runs.json and logs/run-<run-id>-failed-jobs.json.
steps:
  - name: Set up Python
    uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405  # v6.2.0
    with:
      python-version: '3.11'
  - name: Install PyGithub
    run: python -m pip install --quiet PyGithub
  - name: Download CI failure logs
    shell: python
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      REPO: ${{ github.repository }}
      PR_NUMBER: ${{ github.event.issue.number }}
      RUN_ID: ${{ github.event.workflow_run.id || github.event.inputs.run_id }}
    run: |
      import glob
      import json
      import os
      import re
      import sys
      import urllib.request

      from github import Auth, Github

      LOG_DIR = "/tmp/gh-aw/agent/ci-doctor/logs"
      FILTERED_DIR = "/tmp/gh-aw/agent/ci-doctor/filtered"
      SUMMARY_FILE = "/tmp/gh-aw/agent/ci-doctor/summary.txt"
      os.makedirs(LOG_DIR, exist_ok=True)
      os.makedirs(FILTERED_DIR, exist_ok=True)

      HINT_RE = re.compile(
          r"(error[: ]|ERROR|FAIL|panic:|fatal[: ]|undefined[: ]|exception|exit status [^0])"
      )

      repo_name = os.environ.get("REPO", "")
      run_id = (os.environ.get("RUN_ID") or "").strip()
      pr_number = (os.environ.get("PR_NUMBER") or "").strip()
      token = os.environ.get("GH_TOKEN", "")

      github = Github(auth=Auth.Token(token))
      repo = github.get_repo(repo_name)


      def line_count(text):
          return text.count("\n")


      def download_job_log(job):
          """Download the log for a single failed job and pre-locate error hints."""
          log_file = os.path.join(LOG_DIR, f"job-{job.id}.log")
          print(f"Downloading log for job {job.id}...")
          try:
              with urllib.request.urlopen(job.logs_url()) as response:
                  content = response.read().decode("utf-8", "replace")
          except Exception:
              content = "(log download failed)\n"
          with open(log_file, "w", encoding="utf-8") as handle:
              handle.write(content)
          print(f"  -> Saved {line_count(content)} lines to {log_file}")

          hints_file = os.path.join(FILTERED_DIR, f"job-{job.id}-hints.txt")
          matches = []
          for number, line in enumerate(content.splitlines(), start=1):
              if HINT_RE.search(line):
                  matches.append(f"{number}:{line}")
                  if len(matches) >= 30:
                      break
          if matches:
              with open(hints_file, "w", encoding="utf-8") as handle:
                  handle.write("\n".join(matches) + "\n")
              print(f"  -> Pre-located {len(matches)} hint line(s) in {hints_file}")


      def failed_jobs_for_run(workflow_run):
          """Return the failed/cancelled jobs of a run as serialisable dicts."""
          result = []
          for job in workflow_run.jobs():
              if job.conclusion in ("failure", "cancelled"):
                  failed_steps = [step.name for step in (job.steps or []) if step.conclusion == "failure"]
                  result.append({"id": job.id, "name": job.name, "failed_steps": failed_steps})
          return result


      def write_summary_footer(handle):
          """Append the shared 'downloaded files' + 'hint files' footer to the summary."""
          handle.write("\n")
          handle.write(f"Downloaded job log files ({LOG_DIR}):\n")
          for log_file in sorted(glob.glob(os.path.join(LOG_DIR, "job-*.log"))):
              with open(log_file, encoding="utf-8") as fh:
                  count = line_count(fh.read())
              handle.write(f"  {log_file} ({count} lines)\n")
          handle.write("\n")
          handle.write(f"Filtered hint files ({FILTERED_DIR}):\n")
          for hints_file in sorted(glob.glob(os.path.join(FILTERED_DIR, "*-hints.txt"))):
              with open(hints_file, encoding="utf-8") as fh:
                  lines = fh.read().splitlines()
              if not lines:
                  continue
              handle.write(f"  {hints_file} ({len(lines)} matches)\n")
              for line in lines[:3]:
                  handle.write(f"    {line}\n")


      if run_id:
          # ---- run mode: a single workflow run (CI Doctor — Merge Queue) ----
          print(f"=== CI Doctor: Pre-downloading logs for run {run_id} ===")

          workflow_run = repo.get_workflow_run(int(run_id))
          failed_jobs = failed_jobs_for_run(workflow_run)
          with open(os.path.join(LOG_DIR, "failed-jobs.json"), "w", encoding="utf-8") as handle:
              json.dump(failed_jobs, handle, indent=2)

          failed_count = len(failed_jobs)
          print(f"Found {failed_count} failed job(s)")

          with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
              handle.write("=== CI Doctor Pre-Analysis ===\n")
              handle.write(f"Run ID: {run_id}\n")
              handle.write("\n")
              handle.write(f"Failed jobs (details in {LOG_DIR}/failed-jobs.json):\n")
              for job in failed_jobs:
                  handle.write(f"  Job {job['id']}: {job['name']}\n")
                  handle.write(f"    Failed steps: {', '.join(job['failed_steps'])}\n")

          if failed_count == 0:
              print("No failed jobs found, skipping log download")
          else:
              print(json.dumps(failed_jobs, indent=2))
              for job in workflow_run.jobs():
                  if job.conclusion in ("failure", "cancelled"):
                      download_job_log(job)

          with open(SUMMARY_FILE, "a", encoding="utf-8") as handle:
              write_summary_footer(handle)

      elif pr_number:
          # ---- pr mode: every failed run on a pull request head commit ----
          print(f"=== CI Doctor: Pre-downloading logs for PR #{pr_number} ===")

          try:
              head_sha = repo.get_pull(int(pr_number)).head.sha
          except Exception:
              head_sha = ""
          if not head_sha:
              print("Could not resolve a pull request head SHA (is this a PR comment?), skipping log download")
              with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
                  handle.write("No pull request context available.\n")
              sys.exit(0)
          print(f"PR head SHA: {head_sha}")

          # Find all workflow runs for the PR head SHA that failed or were cancelled.
          failed_runs = []
          try:
              for workflow_run in repo.get_workflow_runs(head_sha=head_sha):
                  if workflow_run.conclusion in ("failure", "cancelled"):
                      failed_runs.append({
                          "id": workflow_run.id,
                          "name": workflow_run.name,
                          "url": workflow_run.html_url,
                          "conclusion": workflow_run.conclusion,
                      })
          except Exception:
              failed_runs = []

          # De-duplicate by workflow name, keeping the most recent (highest id) run per workflow.
          latest_by_name = {}
          for entry in failed_runs:
              current = latest_by_name.get(entry["name"])
              if current is None or entry["id"] > current["id"]:
                  latest_by_name[entry["name"]] = entry
          failed_runs = list(latest_by_name.values())

          with open(os.path.join(LOG_DIR, "failed-runs.json"), "w", encoding="utf-8") as handle:
              json.dump(failed_runs, handle, indent=2)

          failed_count = len(failed_runs)
          print(f"Found {failed_count} failed pipeline(s) on PR #{pr_number}")

          with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
              handle.write(f"=== CI Doctor Pre-Analysis (PR #{pr_number}, head {head_sha}) ===\n")
              handle.write("\n")
              handle.write(f"Failed pipelines (details in {LOG_DIR}/failed-runs.json):\n")
              for entry in failed_runs:
                  handle.write(f"  Run {entry['id']}: {entry['name']} [{entry['conclusion']}] {entry['url']}\n")

          if failed_count == 0:
              print("No failed pipelines found, skipping log download")
          else:
              print(json.dumps(failed_runs, indent=2))
              for entry in failed_runs:
                  workflow_run = repo.get_workflow_run(int(entry["id"]))
                  run_failed_jobs = failed_jobs_for_run(workflow_run)
                  run_jobs_path = os.path.join(LOG_DIR, f"run-{entry['id']}-failed-jobs.json")
                  with open(run_jobs_path, "w", encoding="utf-8") as handle:
                      json.dump(run_failed_jobs, handle, indent=2)
                  for job in workflow_run.jobs():
                      if job.conclusion in ("failure", "cancelled"):
                          download_job_log(job)

          with open(SUMMARY_FILE, "a", encoding="utf-8") as handle:
              write_summary_footer(handle)

      else:
          print("Neither RUN_ID nor PR_NUMBER is set; nothing to pre-download.")
          with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
              handle.write("No CI Doctor context (no RUN_ID or PR_NUMBER).\n")

      print("")
      print(f"Pre-analysis complete. Agent should start with {SUMMARY_FILE}")
---

<!--
Shared CI Doctor pre-analysis step. This file has no `on:` trigger, so it is a
shared workflow component: it is imported (never compiled standalone) via

    imports:
      - shared/agentic-workflows/download-failure-logs.md

Imported `steps:` are prepended to the importing workflow's own steps at compile
time. See https://github.github.com/gh-aw/reference/imports/#importing-steps
-->
