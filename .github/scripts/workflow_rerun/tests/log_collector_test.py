# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
log collector tests
"""

import os
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import requests
from github import Github, Auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from workflow_rerun.log_collector import collect_logs_for_run


class LogCollectorTest(unittest.TestCase):
    """
    A class for testing log collection
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=3,
            backoff_jitter=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        cls.session.mount("https://github.com", HTTPAdapter(max_retries=retry_strategy))

        cls.github = Github(auth=Auth.Token(token=os.environ.get('GITHUB_TOKEN')))
        gh_repo = cls.github.get_repo(full_name_or_id='openvinotoolkit/openvino')

        # Looking for reference workflow runs.
        # Their "created_at" time should be within 60 days - the log retention window
        oldest_allowed_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        cls.successful_workflow_run = gh_repo.get_workflow_runs(status='success',
                                                                created=f">={oldest_allowed_date}")[0]
        print(f'Successful workflow run for testing: {cls.successful_workflow_run}', flush=True)

        cls.failed_workflow_run = gh_repo.get_workflow_runs(status='failure',
                                                            created=f">={oldest_allowed_date}")[0]
        print(f'Failed workflow run for testing: {cls.failed_workflow_run}', flush=True)

    def setUp(self):
        print(f'\nIn test: "{self._testMethodName}"', flush=True)

    def test_failed_logs_are_collected(self) -> None:
        """
        Ensure only logs for failed jobs are collected
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            collect_logs_for_run(run=self.failed_workflow_run, logs_dir=logs_dir, session=self.session)
            self.assertTrue(any(logs_dir.iterdir()),
                            'Logs directory should not be empty for failed runs')

    def test_successful_logs_are_not_collected(self) -> None:
        """
        Ensure logs for successful jobs are not collected
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            collect_logs_for_run(run=self.successful_workflow_run, logs_dir=logs_dir, session=self.session)
            self.assertFalse(any(logs_dir.iterdir()),
                            'Logs directory should be empty for successful runs')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.github.close()
