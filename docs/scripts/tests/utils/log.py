# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Doxygen and Sphinx logs parsing routines
"""

import re
from pathlib import Path


class LogParser:
    """
    This class reads a log file and converts it to a structured format represented as a python `dict`
    """
    exclude_symbols = (
        '\x1b[91m',
        '[39;49;00m'
    )

    # a regex that is used to match log lines containing a filepath,
    # a line number, and an error,warning or critical
    regex = r'^(?!\*)(.*?):?([0-9]*):? ?(warning|error|critical): (.+)'

    def __init__(self, log: Path, strip: str, xfail_list: list, suppress_warnings: list):
        """
        Initialize a LogParser object for parsing doxygen and sphinx logs
        :param log: Path to a log file represented as a `pathlib.Path` object
        :param strip: A part of the filepath that should be removed
        :param suppress_warnings: A list of warnings that should be ignored
        :param xfail_list: A list of filepaths that should be ignored
        """
        self.log = log
        if not strip.endswith('/'):
            strip = strip + '/'
        self.strip = strip.replace('\\', '/').lower()
        self.xfail_list = xfail_list
        self.suppress_warnings = suppress_warnings
        self.out = dict()

    def get_match(self, line: str):
        """
        Match a log line against the regex defined by this class
        """
        return re.match(self.regex, line)

    def preprocess_line(self, line):
        """
        Clear log line from unwanted symbols
        """
        for sym in self.exclude_symbols:
            line = line.replace(sym, '')
        return line.strip().lower()

    def strip_path(self, path):
        """
        Strip `path` components ends on `strip`
        """
        path = path.replace('\\', '/').lower()

        new_path = path.split(self.strip)[-1]
        if new_path.startswith('build/docs/'):
            new_path = new_path.split('build/docs/')[-1]
        return new_path

    def filter(self):
        """
        Filter out a log file to remove files or warning based on the values provided in `strip`,
        `suppress_warnings`, and 'xfail_list`
        """
        filtered_out = dict()
        for filepath, warnings in self.out.items():
            filepath = self.strip_path(filepath)
            if filepath in self.xfail_list:
                continue
            warnings = list(filter(lambda item: not self.is_suppressed(item), warnings))
            if warnings:
                filtered_out[filepath] = warnings
        return filtered_out

    def is_suppressed(self, line):
        return any([re.search(re.compile(warning, re.IGNORECASE), line) for warning in self.suppress_warnings])

    def parse(self):
        """
        Parse a log file to convert it to a structured format
        """
        with open(self.log, 'r', errors='ignore') as f:
            log_lines = f.readlines()

        # iterate each line in the log file
        i = 0
        while i < len(log_lines):
            j = i + 1
            line = self.preprocess_line(log_lines[i])
            match = self.get_match(line)
            # if match is true then we found a line containing a filepath,
            # a line number, and a warning/error
            if match and not self.is_suppressed(line):
                filepath = match.group(1) or 'warning'
                linenum = match.group(2)
                warning = match.group(4)
                if not filepath in self.out:
                    self.out[filepath] = set()
                if linenum:
                    warning = f'{warning} line ({linenum})'
                self.out[filepath].add(warning)
                # in this case, the filepath might contain several errors on separate lines,
                # so we need to iterate next lines until we find a line
                # that matches the regex defined in this class
                while j < len(log_lines):
                    next_line = self.preprocess_line(log_lines[j])
                    match = self.get_match(next_line)
                    if match:
                        break
                    if next_line:
                        self.out[filepath].add(self.preprocess_line(next_line))
                    j += 1
            i = j
        return self.out
