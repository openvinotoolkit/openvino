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
import shutil
from unittest.mock import patch, MagicMock

import requests
from github import Github, Auth
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from workflow_rerun.log_analyzer import LogAnalyzer
from workflow_rerun.log_collector import collect_logs_for_run
from workflow_rerun.rerunner import analyze_and_rerun


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
        cls.test_logs_with_error_dir = cls._cwd.joinpath('data', 'logs_with_error')

        cls.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=3,
            backoff_jitter=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        cls.session.mount("https://github.com", HTTPAdapter(max_retries=retry_strategy))

        # Only create a GitHub client/run if token is available (otherwise tests should be offline)
        cls.github = None
        cls.wf_run = None
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            cls.github = Github(auth=Auth.Token(token=token))
            gh_repo = cls.github.get_repo(full_name_or_id='openvinotoolkit/openvino')

            oldest_allowed_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            cls.wf_run = gh_repo.get_workflow_runs(status='failure',
                                                   created=f">={oldest_allowed_date}")[0]
            print(f'Workflow run for testing: {cls.wf_run}', flush=True)

    def setUp(self):
        print(f'\nIn test: "{self._testMethodName}"', flush=True)

    @unittest.skipUnless(os.environ.get('GITHUB_TOKEN'), 'GITHUB_TOKEN not set; skipping live GitHub integration test')
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

    def test_analyze_and_rerun_records_to_db_offline(self) -> None:
        """Offline integration-style test: uses local logs and mocks all network/DB side effects."""

        def fake_collect_logs_for_run(*, run, logs_dir: Path, session):
            # Populate the temp logs_dir with our checked-in test logs.
            for p in self.test_logs_with_error_dir.rglob('*'):
                if p.is_file():
                    rel = p.relative_to(self.test_logs_with_error_dir)
                    dst = logs_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)

        mock_run = MagicMock()
        mock_run.html_url = 'https://github.com/example/repo/actions/runs/123'

        mock_session = MagicMock()

        repository_name = 'openvinotoolkit/openvino'
        run_id = 123
        rerunner_run_id = 456

        with patch('workflow_rerun.rerunner.collect_logs_for_run', side_effect=fake_collect_logs_for_run) as collect_mock, \
             patch('workflow_rerun.rerunner.rerun_failed_jobs') as rerun_mock, \
             patch('workflow_rerun.rerunner.record_rerun_to_db') as record_mock:
            analyze_and_rerun(
                run=mock_run,
                repository_name=repository_name,
                run_id=run_id,
                rerunner_run_id=rerunner_run_id,
                errors_file=self.errors_to_look_for_file,
                is_dry_run=False,
                session=mock_session
            )

            collect_mock.assert_called_once()
            rerun_mock.assert_called_once_with(repository_name, run_id, mock_session)
            record_mock.assert_called_once()

            # Basic sanity on record_rerun_to_db args
            args = record_mock.call_args[0]
            self.assertEqual(args[0], repository_name)
            self.assertEqual(args[1], run_id)
            self.assertEqual(args[3], rerunner_run_id)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.github is not None:
            cls.github.close()
