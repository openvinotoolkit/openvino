# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Doxygen log parsing routines
"""
from collections import defaultdict
import argparse
import re


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doxygen', type=str, required=True, default=None, help='Path to doxygen.log file')
    parser.add_argument('--doxygen-strip', type=str, required=False, default='tmp_docs/', help='Path to doxygen.log file')
    return parser.parse_args()


def strip_timestmp(text):
    """Strip jenkins timestamp
    """
    return text.split(']')[-1]


def strip_path(path, strip):
    """Strip `path` components ends on `strip`
    """
    strip = strip.replace('\\', '/')
    if not strip.endswith('/'):
        strip = strip + '/'
    new_path = path.split(strip)[-1]
    if new_path.startswith('build/docs/'):
        new_path = new_path.split('build/docs/')[-1]
    return new_path


def _get_file_line(text):
    """Extracts file and line from Doxygen warning line
    """
    if text:
        location = text.split()[-1]
        file_line = location.rsplit(':', 1)
        if len(file_line) == 2:
            return file_line
    return '', ''


def parse(log, strip):
    """Extracts {file: errors} from doxygen log
    """
    log = log.splitlines()
    files = defaultdict(lambda: set())  # pylint: disable=unnecessary-lambda
    idx = 0
    prev_file = ''
    prev_line = ''
    while idx < len(log):  # pylint: disable=too-many-nested-blocks
        try:
            log_line = strip_timestmp(log[idx]).strip()
            processing_verb = next(
                filter(log_line.startswith,
                       ('Reading /', 'Parsing file /', 'Preprocessing /')),
                None)
            if processing_verb:
                files[strip_path(log_line[len(processing_verb) - 1:-3],
                                 strip)] = set()
            elif 'warning:' in log_line:
                warning = list(map(str.strip, log_line.split(': warning:')))
                file, line = _get_file_line(warning[0])
                file = strip_path(file, strip)
                if len(warning) == 1:
                    file = prev_file
                    line = prev_line
                    error = warning[0]
                else:
                    error = warning[1]
                if error.endswith(':'):
                    continuation = []
                    while idx + 1 < len(log):
                        peek = strip_timestmp(log[idx + 1])
                        if not peek.startswith('  '):
                            break
                        continuation += [peek]
                        idx += 1
                    error += ';'.join(continuation)
                if line:
                    error = '{error} (line: {line})'.format(
                        line=line, error=error)
                if not file or 'deprecated' in file:
                    files['doxygen_errors'].update([error])
                else:
                    prev_file = file
                    prev_line = line
                    files[file].update([error])
            elif log_line.startswith('explicit link request') and 'in layout file' in log_line:
                match = re.search(r"\'(.+?)\'", log_line)
                if match:
                    file = match.group(1)
                    files[file].update([log_line])
                else:
                    files['doxygen_errors'].update([log_line])
            idx += 1
        except:
            print('Parsing error at line {}\n\n{}\n'.format(idx, log[idx]))
            raise
    return files


if __name__ == '__main__':
    arguments = parse_arguments()
    with open(arguments.doxygen, 'r') as log:
        files = parse(log.read(), arguments.doxygen_strip)
