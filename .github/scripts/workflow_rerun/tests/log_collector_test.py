# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
log collector tests
"""

import os
import unittest
import tempfile
from pathlib import Path
from datetime import datetime

from github import Github, Auth

from workflow_rerun.log_collector import collect_logs_for_run


class LogCollectorTest(unittest.TestCase):
    """
    A class for testing log collection
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self._cwd = Path(__file__).parent
        self.github = Github(auth=Auth.Token(token=os.environ.get('GITHUB_TOKEN')))
        self.gh_repo = self.github.get_repo(full_name_or_id='openvinotoolkit/openvino')
        # Use the logs of the most recent successfull pipeline
        # Its "created_at" time should be within 60 days - the log retention window
        self.wf_run = None
        for run in self.gh_repo.get_workflow_runs(status='success'):
            if (datetime.now(run.created_at.tzinfo) - run.created_at).days < 45:
                self.wf_run = run
                break
        if not self.wf_run:
            raise RuntimeError('No suitable workflow run found for testing')
        print(f'Workflow run for testing: {self.wf_run}', flush=True)

    def test_log_collection(self) -> None:
        """
        Ensure log collection is working
        """
        log_archive_path = Path(tempfile.NamedTemporaryFile(suffix='.zip').name)
        collect_logs_for_run(run=self.wf_run, log_archive_path=log_archive_path)
        self.assertTrue(Path(log_archive_path).exists())

    def tearDown(self) -> None:
        self.github.close()
