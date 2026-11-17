# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gate automatic CI Doctor investigations on a failed Overall_Status job."""

from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import urlsplit

from common import handle_numeric_id, require_env, resolve_repository


def resolve_target(repository: str, run_id: str, link: str, attempt: str) -> tuple[str, str, str]:
    """Resolve a run ID or a manual run link without accepting arbitrary API paths."""
    if not run_id and link:
        url = urlsplit(link)
        server = urlsplit(os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
        match = re.fullmatch(
            r"/([^/]+/[^/]+)/actions/runs/([0-9]+)(?:/attempts/([0-9]+))?/?",
            url.path,
        )
        if url.scheme != "https" or url.netloc != server.netloc or not match:
            raise SystemExit("link must be a GitHub Actions workflow run URL")
        repository, run_id, attempt = match.group(1), match.group(2), match.group(3) or ""
    repository = resolve_repository(repository)
    if run_id:
        handle_numeric_id(run_id, "run_id")
    if attempt:
        handle_numeric_id(attempt, "run_attempt")
    return repository, run_id, attempt


def overall_status_failed(repository: str, run_id: str, attempt: str) -> bool:
    """Inspect all job pages, restricted to the triggering attempt when supplied."""
    endpoint = f"repos/{repository}/actions/runs/{run_id}"
    if attempt:
        endpoint += f"/attempts/{attempt}"
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", f"{endpoint}/jobs?filter=latest&per_page=100"],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    jobs = [job for page in json.loads(result.stdout) for job in page["jobs"]]
    overall_jobs = [
        job for job in jobs
        if re.fullmatch(r"ci/gha_overall_status(?:_[A-Za-z0-9_.-]+)?", job["name"])
    ]
    for job in overall_jobs:
        print(f"{job['name']}: {job['conclusion']}")
    if not overall_jobs:
        print("No Overall_Status job found.")
    return any(job["conclusion"] == "failure" for job in overall_jobs)


def main() -> None:
    repository, run_id, attempt = resolve_target(
        require_env("REPO"),
        os.environ.get("RUN_ID", "").strip(),
        os.environ.get("LINK", "").strip(),
        os.environ.get("RUN_ATTEMPT", "").strip(),
    )
    investigate = bool(run_id) and overall_status_failed(repository, run_id, attempt)
    if not investigate:
        print("Skipping investigation: no failed Overall_Status job.")
    with open(require_env("GITHUB_OUTPUT"), "a", encoding="utf-8") as handle:
        handle.write(f"should_investigate={str(investigate).lower()}\n")


if __name__ == "__main__":
    main()
