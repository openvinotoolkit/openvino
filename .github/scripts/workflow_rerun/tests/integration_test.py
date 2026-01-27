# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests
"""

import unittest
from pathlib import Path
from datetime import datetime, timedelta
import os
import tempfile

import requests
from github import Github, Auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from workflow_rerun.log_analyzer import LogAnalyzer
from workflow_rerun.log_collector import collect_logs_for_run


class IntegrationTest(unittest.TestCase):
    """
    A class for testing integration between LogAnalyzer and log_collection
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._cwd = Path(__file__).parent
        cls.errors_to_look_for_file = cls._cwd.parent.joinpath(
            'errors_to_look_for.json'
        )

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

        # Even if we use "failure" for status we cannot guarantee logs containing any of the known error
        # So these tests use the logs of the most recent failed pipeline
        # Its "created_at" time should be within 60 days - the log retention window
        oldest_allowed_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        cls.wf_run = gh_repo.get_workflow_runs(status='failure',
                                               created=f">={oldest_allowed_date}")[0]
        print(f'Workflow run for testing: {cls.wf_run}', flush=True)

    def setUp(self):
        print(f'\nIn test: "{self._testMethodName}"', flush=True)

    def test_log_collection_and_analysis(self) -> None:
        """
        Ensure logs collected by collect_logs_for_run are analyzed by LogAnalyzer
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            collect_logs_for_run(run=self.wf_run,
                                 logs_dir=logs_dir,
                                 session=self.session)

            analyzer = LogAnalyzer(
                path_to_logs=logs_dir,
                path_to_errors_file=self.errors_to_look_for_file,
            )
            self.assertTrue(len(analyzer._log_files) > 0,
                            'Failed run log files should be collected for failed jobs')
            analyzer.analyze()
            if analyzer.found_matching_error:
                print(f'Found matching error, ticket: {analyzer.found_error_ticket}')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.github.close()
