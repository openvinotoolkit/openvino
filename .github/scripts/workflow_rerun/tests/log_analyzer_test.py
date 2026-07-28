# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LogAnalyzer tests
"""

import unittest
from pathlib import Path


from workflow_rerun.log_analyzer import CI_DOCTOR_PATTERN_TICKET, LogAnalyzer


class LogAnalyzerTest(unittest.TestCase):
    """
    A class for testing LogAnalyzer
    """

    def setUp(self) -> None:
        print(f'\nIn test: "{self._testMethodName}"', flush=True)
        self._cwd = Path(__file__).parent
        self.logs_dir_with_error = self._cwd.joinpath("data").joinpath(
            'logs_with_error'
        )
        self.logs_dir_wo_error = self._cwd.joinpath("data").joinpath(
            'logs_wo_error'
        )
        self.errors_to_look_for_file = self._cwd.parent.joinpath(
            'errors_to_look_for.json'
        )
        self.empty_errors_file = self._cwd.joinpath('data').joinpath(
            'empty_errors.json'
        )
        self.patterns_dir = self._cwd.joinpath('data').joinpath('patterns')

    def test_log_analyzer_instantiation(self) -> None:
        """
        Ensure LogAnalyzer is instantiated correctly.
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
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
            path_to_logs=self.logs_dir_wo_error,
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
            path_to_logs=self.logs_dir_with_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertTrue(analyzer.found_matching_error)
        self.assertEqual(analyzer.found_error_ticket, 130955)
        self.assertEqual(analyzer.matched_error_text,
                         'Network is unreachable')

    def test_analyzer_wo_error(self) -> None:
        """
        Ensure LogAnalyzer does not find an error in the log files w/o errors
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.errors_to_look_for_file,
        )
        analyzer.analyze()
        self.assertFalse(analyzer.found_matching_error)

    def test_patterns_loaded_from_directory(self) -> None:
        """
        Ensure LogAnalyzer loads rerun_search_string values from CI Doctor MQ
        pattern files (tagged with the CI_DOCTOR_PATTERN_TICKET sentinel ticket)
        and skips patterns whose rerun_search_string is null.
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.empty_errors_file,
            patterns_dir=self.patterns_dir,
        )

        pattern_errors = analyzer._errors_to_look_for
        self.assertEqual(
            len(pattern_errors),
            1,
            'Only the pattern with a non-null rerun_search_string should be loaded',
        )
        self.assertEqual(pattern_errors[0]['error_text'], 'label empty or too long')
        self.assertEqual(
            pattern_errors[0]['ticket'],
            CI_DOCTOR_PATTERN_TICKET,
            'Pattern-derived errors must carry the CI_DOCTOR_PATTERN_TICKET sentinel',
        )

    def test_analyzer_matches_pattern_search_string(self) -> None:
        """
        Ensure LogAnalyzer can find an error using a rerun_search_string coming
        from a CI Doctor MQ pattern file, reporting the sentinel ticket for it.
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_with_error,
            path_to_errors_file=self.empty_errors_file,
            patterns_dir=self.patterns_dir,
        )
        analyzer.analyze()
        self.assertTrue(analyzer.found_matching_error)
        self.assertEqual(analyzer.found_error_ticket, CI_DOCTOR_PATTERN_TICKET)
        self.assertEqual(analyzer.matched_error_text, 'label empty or too long')

    def test_missing_patterns_dir_is_ignored(self) -> None:
        """
        Ensure a non-existent patterns_dir does not break analysis and simply
        adds no pattern-derived errors.
        """
        analyzer = LogAnalyzer(
            path_to_logs=self.logs_dir_wo_error,
            path_to_errors_file=self.empty_errors_file,
            patterns_dir=self._cwd.joinpath('data').joinpath('does_not_exist'),
        )
        self.assertEqual(analyzer._errors_to_look_for, [])
        analyzer.analyze()
        self.assertFalse(analyzer.found_matching_error)
