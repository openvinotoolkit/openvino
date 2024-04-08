"""
LogAnalyzer tests
"""
from pathlib import Path
import unittest
from workflow_rerun.log_analyzer import LogAnalyzer


class LogAnalyzerTest(unittest.TestCase):
    """
    A class for testing LogAnalyzer
    """
    
    def setUp(self) -> None:
        self._cwd = Path(__file__).parent
        self.log_archive_with_error = self._cwd.joinpath('data').joinpath('log_archive_wo_error.zip')
        self.log_archive_wo_error = self._cwd.joinpath('data').joinpath('log_archive_with_error.zip')
        self.errors_to_look_for_file = self._cwd.joinpath('data').joinpath('errors_to_look_for.json')
    
    def test_log_analyzer_instantiation(self) -> None:
        """
        Ensure LogAnalyzer is instantiated correctly.
        """
        analyzer = LogAnalyzer(path_to_log_archive=self.log_archive_wo_error,
                               path_to_errors_file=self.errors_to_look_for_file) 
        self.assertTrue(hasattr(analyzer, '_errors_to_look_for'),
                        "Analyzer should have _errors_to_look_for")
        self.assertTrue(hasattr(analyzer, '_log_files'),
                        "Analyzer should have _log_files")
        
        for error_data in analyzer._errors_to_look_for:
            self.assertTrue(error_data['error_text'], 'Each error_data should have text')
            self.assertTrue(error_data['ticket'], 'Each error_data should have ticket')
        
        for log_file in analyzer._log_files:
            self.assertTrue(log_file['file_name'], 'Each log_file should have file_name')
            self.assertTrue(log_file['path'], 'Each log_file should have path')
        

    def test_log_parsing(self) -> None:
        """
        Ensure log parsing function returns correct results
        """
        analyzer = LogAnalyzer(path_to_log_archive=self.log_archive_wo_error,
                               path_to_errors_file=self.errors_to_look_for_file)
        
        data = ('Connection was reset', 'Failed to connect to github.com', 'Could not resolve host: github.com')
        expected = ('connection was reset', 'failed to connect to github com', 'could not resolve host github com')
        
        for input_str, expected_str in zip(data, expected):
            self.assertEqual(analyzer._clean_up_string(string=input_str), expected_str)
