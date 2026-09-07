# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for live-GitHub integration tests
"""

from typing import Optional

from github.Repository import Repository
from github.WorkflowRun import WorkflowRun


def find_failed_run_with_logs(gh_repo: Repository,
                              created_filter: str,
                              max_candidates: int = 20) -> Optional[WorkflowRun]:
    """
    Some workflow runs reported as "failure" never actually start any jobs
    (e.g. the workflow file itself is invalid), so they have no logs to
    download/analyze. This looks through the most recent failed runs and
    returns the first one that has at least one failed job, or None if no
    such run is found among the checked candidates.
    """
    candidate_runs = gh_repo.get_workflow_runs(status='failure', created=created_filter)
    for candidate_run in candidate_runs[:max_candidates]:
        if any(job.conclusion == 'failure' for job in candidate_run.jobs()):
            return candidate_run
    return None
