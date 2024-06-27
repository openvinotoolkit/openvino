# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import requests
from github.WorkflowRun import WorkflowRun
from workflow_rerun.constants import GITHUB_TOKEN, LOGGER


def collect_logs_for_run(run: WorkflowRun,
                         log_archive_path: Path) -> Path:
    """
    Collects log archive for a pipeline
    """
    with open(file=log_archive_path, 
              mode='wb') as log_archive:
        LOGGER.info(f'STARTED LOG COLLECTION FOR {run.id} IN {log_archive_path}')
        # PyGitHub does not expose the "/repos/{owner}/{repo}/actions/runs/{run_id}/logs" endpoint so we have to use requests
        log_archive.write(requests.get(url=run.logs_url, 
                                       headers={'Authorization': f'Bearer {GITHUB_TOKEN}'}).content)
        LOGGER.info(f'COLLECTED LOGS FOR {run.id} IN {log_archive_path}')

    return log_archive_path
