# Copyright (C) 2018-2021 Intel Corporation
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
    path = path.replace('\\', '/').lower()
    strip = strip.replace('\\', '/').lower()
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
        if text.count(':') == 1:
            return text, ''
        location = text.split()[-1]
        file_line = location.rsplit(':', 1)
        if len(file_line) == 2:
            return file_line
    return '', ''


def parse(log, strip, suppress_warnings=list()):
    """Extracts {file: errors} from doxygen log
    """
    log = log.splitlines()
    files = defaultdict(lambda: set())  # pylint: disable=unnecessary-lambda
    idx = 0
    prev_file = ''
    while idx < len(log):  # pylint: disable=too-many-nested-blocks
        try:
            errors = []
            log_line = strip_timestmp(log[idx]).strip().lower()
            # remove unnecessary characters
            log_line = log_line.replace('\x1b[91m', '')
            processing_verb = next(
                filter(log_line.startswith,
                       ('Reading /', 'Parsing file /', 'Preprocessing /')),
                None)
            if processing_verb:
                files[strip_path(log_line[len(processing_verb) - 1:-3],
                                 strip)] = set()
            elif 'warning:' in log_line:
                warning = list(map(str.strip, log_line.split(': warning:')))
                if len(warning) == 1:
                    file = prev_file
                    error = warning[0]
                else:
                    file = warning[0]
                    error = warning[1]
                error = error.replace('[39;49;00m', '')
                file, line = _get_file_line(file)
                file = strip_path(file, strip)
                errors.append(error)
                if error.endswith(':'):
                    while idx + 1 < len(log):
                        peek = strip_timestmp(log[idx + 1]).replace('[39;49;00m', '')
                        if not peek.startswith('  '):
                            break
                        errors.append(peek)
                        idx += 1

                errors = list(filter(lambda item: not any([re.match(warning, item) for warning in suppress_warnings]), errors))

                if not errors:
                    idx += 1
                    continue

                if line:
                    errors[0] = '{error} (line: {line})'.format(
                        line=line, error=errors[0])
                if not file or 'deprecated' in file:
                    files['doxygen_errors'].update(errors)
                else:
                    prev_file = file
                    files[file].update(errors)
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
