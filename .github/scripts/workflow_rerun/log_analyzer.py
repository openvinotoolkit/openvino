# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import re
import tempfile
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

from workflow_rerun.constants import LOGGER


class LogFile(TypedDict):
    file_name: str
    path: Path


class ErrorData(TypedDict):
    error_text: str
    ticket: int


class LogAnalyzer:
    def __init__(self,
                 path_to_log_archive: Path,
                 path_to_errors_file: Path) -> None:
        self._path_to_log_archive = path_to_log_archive
        self._path_to_errors_file = path_to_errors_file

        self._errors_to_look_for: list[ErrorData] = []
        self._collect_errors_to_look_for()

        self._log_dir = tempfile.TemporaryDirectory().name

        self._log_files: list[LogFile] = []
        self._collect_log_files()
        all_txt_log_files_pretty = '\n'.join(map(lambda item: str(item['path']), self._log_files))
        LOGGER.info(f'ALL .txt LOG FILES: \n{all_txt_log_files_pretty}')

        self.found_matching_error = False
        self.found_error_ticket = None

    def _collect_errors_to_look_for(self) -> None:
        with open(file=self._path_to_errors_file,
                  mode='r',
                  encoding='utf-8') as errors_file:
            errors_data = json.load(errors_file)
            for error_data in errors_data:
                self._errors_to_look_for.append(
                    ErrorData(error_text=error_data['error_text'], 
                              ticket=error_data['ticket'])
                    )

    def _collect_log_files(self) -> None:
        """
        Collects the .txt log files from the log archive

        The GitHub Actions pipeline logs archive should have the following structure:
            > Job_name_0
                > step_name_0.txt
                > step_name_1.txt
                ...
            > Job_name_1
                > step_name_0.txt
                > step_name_1.txt
                ...
            > Job_name_2
                ...
            ...
        
        We need to only analyze the `*.txt` files
        """

        with ZipFile(file=self._path_to_log_archive,
                     mode='r') as zip_file:
            zip_file.extractall(self._log_dir)

        for _file in Path(self._log_dir).iterdir():
            if _file.is_dir():
                for log_file in _file.iterdir():
                    self._log_files.append(LogFile(file_name=log_file.name,
                                                   path=log_file.resolve()))

    def _is_error_in_log(self,
                         error_to_look_for: str,
                         log_file_path: Path) -> bool:
        """
        Searches for the error in the provided log
        """

        error_to_look_for = self._clean_up_string(error_to_look_for)

        with open(file=log_file_path,
                  mode='r',
                  encoding='utf-8') as log_file:
            for line in log_file:
                if error_to_look_for in self._clean_up_string(line):
                    return True
        return False

    @staticmethod
    def _clean_up_string(string: str) -> str:
        """
        Replaces special characters with spaces in the string, strips it from leading and following spaces,
        and lowers it
        
        for "Could not resolve host: github.com" returns "could not resolve host github com"
        
        This cleanup is applied to both errors to look for and logs themselves for matching
        """
        return re.sub(r'[^A-Za-z0-9]+', ' ', string).lower().strip()

    def analyze(self) -> None:
        """
        Iterates over the known errors and tries to find them in the collected log files
        """
        for error in self._errors_to_look_for:

            LOGGER.info(f'LOOKING FOR "{error["error_text"]}" ERROR...')

            for log_file in self._log_files:
                if self._is_error_in_log(error_to_look_for=error['error_text'],
                                         log_file_path=log_file['path']):
                    LOGGER.info(f'FOUND "{error["error_text"]}" ERROR IN {log_file["path"]}. TICKET: {error["ticket"]}')
                    self.found_matching_error = True
                    self.found_error_ticket = error['ticket']
                    return


if __name__ == '__main__':
    # Usage example
    log_analyzer = LogAnalyzer(path_to_log_archive=Path('/tmp/logs/log.zip'),
                               path_to_errors_file=Path('/tmp/errors_to_look_for.json'))
    log_analyzer.analyze()
    if log_analyzer.found_matching_error:
        print('found matching error, see logs above')
