# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LogAnalyzer tests
"""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory


from workflow_rerun.log_analyzer import NO_CATEGORY, LogAnalyzer


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
            self.assertTrue(error_data['category'], 'Each error_data should have category')

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
        self.assertEqual(analyzer.matched_category, 'Network')

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

    def test_passing_test_name_does_not_trigger_rerun(self) -> None:
        with TemporaryDirectory() as logs_dir:
            Path(logs_dir, 'pytest.txt').write_text(
                'test_keras_lstm_flags_float32 PASSED\n'
                'TestMatrixBandPart::test_matrix_band_part_basic[param] PASSED\n'
                'test_unrelated FAILED\n',
                encoding='utf-8',
            )
            analyzer = LogAnalyzer(
                path_to_logs=Path(logs_dir),
                path_to_errors_file=self.errors_to_look_for_file,
            )
            analyzer.analyze()
            self.assertFalse(analyzer.found_matching_error)

    def test_generic_accuracy_failure_does_not_trigger_rerun(self) -> None:
        with TemporaryDirectory() as logs_dir:
            Path(logs_dir, 'accuracy.txt').write_text(
                'AssertionError: Comparing with Framework failed: ie_res=[1.0]; framework_res=[2.0].\n'
                '[ COMPARATION ] COMPARATION FAILED! abs_threshold: 0.0001 rel_threshold: 0.0001\n',
                encoding='utf-8',
            )
            analyzer = LogAnalyzer(
                path_to_logs=Path(logs_dir),
                path_to_errors_file=self.errors_to_look_for_file,
            )
            analyzer.analyze()
            self.assertFalse(analyzer.found_matching_error)

    def test_unmatchable_placeholder_patterns_are_not_in_database(self) -> None:
        entries = json.loads(self.errors_to_look_for_file.read_text(encoding='utf-8'))
        error_texts = {entry['error_text'] for entry in entries}
        self.assertNotIn('AssertionError: Comparing with Framework failed ie_res framework_res', error_texts)
        self.assertNotIn('Failed: Timeout (>Xs) from pytest-timeout', error_texts)
