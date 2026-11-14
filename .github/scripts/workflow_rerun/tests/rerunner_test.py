# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Rerunner tests
"""

import unittest
from pathlib import Path
from unittest import mock

import requests

from workflow_rerun.rerunner import analyze_and_rerun


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(f'{status_code} Error', response=response)


class RerunnerTest(unittest.TestCase):
    """
    A class for testing the rerunner flow without access to GitHub
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self.run_mock = mock.Mock(html_url='https://github.com/org/repo/actions/runs/1')

    def _analyze_and_rerun(self):
        return analyze_and_rerun(run=self.run_mock,
                                 repository_name='org/repo',
                                 run_id=1,
                                 rerunner_run_id=2,
                                 errors_file=Path('errors.json'),
                                 is_dry_run=False,
                                 session=mock.Mock())

    @mock.patch('workflow_rerun.rerunner.rerun_failed_jobs')
    @mock.patch('workflow_rerun.rerunner.LogAnalyzer')
    @mock.patch('workflow_rerun.rerunner.collect_logs_for_run')
    def test_missing_logs_are_skipped(self, collect_logs, log_analyzer, rerun_failed_jobs) -> None:
        """
        A run whose logs are not found (HTTP 404) is skipped, not retriggered
        """
        collect_logs.side_effect = _http_error(404)

        self._analyze_and_rerun()

        log_analyzer.assert_not_called()
        rerun_failed_jobs.assert_not_called()

    @mock.patch('workflow_rerun.rerunner.LogAnalyzer')
    @mock.patch('workflow_rerun.rerunner.collect_logs_for_run')
    def test_other_http_errors_are_raised(self, collect_logs, log_analyzer) -> None:
        """
        Errors other than "logs not found" must still fail the rerunner
        """
        collect_logs.side_effect = _http_error(403)

        with self.assertRaises(requests.HTTPError):
            self._analyze_and_rerun()

        log_analyzer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
