# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests
"""

import unittest
from pathlib import Path
from github import Github, Auth
import os
import tempfile


from workflow_rerun.log_analyzer import LogAnalyzer
from workflow_rerun.log_collector import collect_logs_for_run


class IntegrationTest(unittest.TestCase):
    """
    A class for testing integration between LogAnalyzer and log_collection
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self._cwd = Path(__file__).parent
        self.errors_to_look_for_file = self._cwd.parent.joinpath(
            'errors_to_look_for.json'
        )
        self.github = Github(auth=Auth.Token(token=os.environ.get('GITHUB_TOKEN')))
        self.gh_repo = self.github.get_repo(full_name_or_id='openvinotoolkit/openvino')

        # Even if we use "failure" for status we cannot guarantee logs containing any of the known error
        # So these tests use the logs of the most recent successfull pipeline
        self.wf_run = self.gh_repo.get_workflow_runs(status='success')[0]
        print(f'Workflow run for testing: {self.wf_run}', flush=True)

    def test_log_collection_and_analysis(self) -> None:
        """
        Ensure logs collected by collect_logs_for_run are analyzed by LogAnalyzer
        """

        log_archive_path = Path(tempfile.NamedTemporaryFile(suffix='.zip').name)
        collect_logs_for_run(run=self.wf_run, 
                             log_archive_path=log_archive_path)

        analyzer = LogAnalyzer(
            path_to_log_archive=log_archive_path,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertFalse(analyzer.found_matching_error)
    
    def tearDown(self) -> None:
        self.github.close()
