# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from zipfile import ZipFile
import tempfile

import requests
from github.WorkflowRun import WorkflowRun
from workflow_rerun.constants import GITHUB_TOKEN, LOGGER


def collect_logs_for_run(run: WorkflowRun,
                         logs_dir: Path) -> Path:
    """
    Downloads logs of a given Workflow Run,
    saves them to a specified path, and returns that path.

    We don't need successful job logs, so we remove them.
    We could've just downloaded logs for failed jobs only,
    but when you download all logs from a workflow run,
    GitHub includes "system.txt" files for each job, which can also
    contain errors on which we might want to trigger rerun.

    Example log archive structure:
    .
    ├── 10_Pytorch Layer Tests _ PyTorch Layer Tests.txt
    ├── 11_CPU functional tests _ CPU functional tests.txt
    ├── 12_C++ unit tests _ C++ unit tests.txt
    ├── 13_OpenVINO tokenizers extension _ OpenVINO tokenizers extension.txt
    ├── C++ unit tests _ C++ unit tests
    │   └── system.txt
    ├── CPU functional tests _ CPU functional tests
    │   └── system.txt
    ├── OpenVINO tokenizers extension _ OpenVINO tokenizers extension
    │   └── system.txt
    ├── Pytorch Layer Tests _ PyTorch Layer Tests
        └── system.txt

    Sometimes though, directories contain log files for each individual step,
    IN ADDITION to the full log in root of the directory:
    .
    ├── 1_Build.txt
    └── Build
        ├── 13_Upload build logs.txt
        ├── 1_Set up job.txt
        ├── 24_Post Clone vcpkg.txt
        ├── 25_Post Clone OpenVINO.txt
        ├── 26_Stop containers.txt
        ├── 27_Complete job.txt
        ├── 2_Initialize containers.txt
        ├── 3_Clone OpenVINO.txt
        ├── 4_Get VCPKG version and put it into GitHub ENV.txt
        ├── 5_Init submodules for non vcpkg dependencies.txt
        ├── 6_Clone vcpkg.txt
        ├── 7_System info.txt
        ├── 8_Build vcpkg.txt
        ├── 9_CMake - configure.txt
        └── system.txt

    In that case, we need only 'system.txt' file from each directory
    """
    # Get failed jobs
    jobs = run.jobs()
    failed_jobs = [job for job in jobs if job.conclusion == 'failure']
    LOGGER.info(f'FAILED JOBS: {[job.name for job in failed_jobs]}')

    with tempfile.NamedTemporaryFile(suffix='.zip') as temp_file:
        log_archive_path = Path(temp_file.name)

        # Download logs archive
        with open(file=log_archive_path,
                mode='wb') as log_archive:
            LOGGER.info(f'DOWNLOADING LOGS FOR RUN ID {run.id}')
            # PyGitHub does not expose the "/repos/{owner}/{repo}/actions/runs/{run_id}/logs" endpoint so we have to use requests
            LOGGER.debug(f'Downloading logs from {run.logs_url}')
            response = requests.get(url=run.logs_url,
                                        headers={'Authorization': f'Bearer {GITHUB_TOKEN}'})
            response.raise_for_status()
            log_archive.write(response.content)

        # Unpack it
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_temp_dir = Path(temp_dir)

            with ZipFile(file=log_archive_path,
                        mode='r') as zip_file:
                zip_file.extractall(logs_temp_dir)

            # Traverse the unpacked logs to find the ones of failed jobs
            for job in failed_jobs:
                job_filename = job.name.replace('/', '_')
                LOGGER.debug(f'Looking for failed job logs with filename: {job_filename}')

                for p in logs_temp_dir.iterdir():
                    # Move failed jobs' logs to the final destination
                    if p.is_dir() and p.name.startswith(job_filename):
                        LOGGER.debug(f'Keeping system.txt from directory {p} for failed job {job.name}')
                        (p / 'system.txt').rename(logs_dir / f'{job_filename}__system.txt')
                    elif p.is_file() and p.name.endswith(job_filename + '.txt'):
                        LOGGER.debug(f'Keeping file {p} for failed job {job.name}')
                        p.rename(logs_dir / p.name)

    LOGGER.info(f'COLLECTED LOGS FOR {run.id} IN {logs_dir}')
    return logs_dir
