# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LogAnalyzer tests
"""

import unittest
from pathlib import Path


from workflow_rerun.log_analyzer import LogAnalyzer


class LogAnalyzerTest(unittest.TestCase):
    """
    A class for testing LogAnalyzer
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self._cwd = Path(__file__).parent
        self.log_archive_with_error = self._cwd.joinpath("data").joinpath(
            'log_archive_with_error.zip'
        )
        self.log_archive_wo_error = self._cwd.joinpath("data").joinpath(
            'log_archive_wo_error.zip'
        )
        self.errors_to_look_for_file = self._cwd.parent.joinpath(
            'errors_to_look_for.json'
        )

    def test_log_analyzer_instantiation(self) -> None:
        """
        Ensure LogAnalyzer is instantiated correctly.
        """
        analyzer = LogAnalyzer(
            path_to_log_archive=self.log_archive_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        self.assertTrue(
            hasattr(analyzer, '_errors_to_look_for'),
            'Analyzer should have _errors_to_look_for',
        )
        self.assertTrue(
            hasattr(analyzer, '_log_files'), 'Analyzer should have _log_files'
        )

        for error_data in analyzer._errors_to_look_for:
            self.assertTrue(
                error_data['error_text'], 'Each error_data should have text'
            )
            self.assertTrue(error_data['ticket'], 'Each error_data should have ticket')

        for log_file in analyzer._log_files:
            self.assertTrue(
                log_file['file_name'], 'Each log_file should have file_name'
            )
            self.assertTrue(log_file['path'], 'Each log_file should have path')

    def test_string_cleanup(self) -> None:
        """
        Ensure log cleanup function returns correct results
        """
        analyzer = LogAnalyzer(
            path_to_log_archive=self.log_archive_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )

        data = (
            'Connection was reset',
            'Failed to connect to github.com',
            'Could not resolve host: github.com',
        )
        expected = (
            'connection was reset',
            'failed to connect to github com',
            'could not resolve host github com',
        )

        for input_str, expected_str in zip(data, expected):
            self.assertEqual(analyzer._clean_up_string(string=input_str), expected_str)

    def test_analyzer_with_error(self) -> None:
        """
        Ensure LogAnalyzer can find an error
        """
        analyzer = LogAnalyzer(
            path_to_log_archive=self.log_archive_with_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertTrue(analyzer.found_matching_error)

    def test_analyzer_wo_error(self) -> None:
        """
        Ensure LogAnalyzer does not find an error in the log files w/o errors
        """
        analyzer = LogAnalyzer(
            path_to_log_archive=self.log_archive_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertFalse(analyzer.found_matching_error)
