# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Configuration for tests.

Tests for documentation utilize pytest test framework for tests execution
and reports generation.

Documentation generation tests process Doxygen log to generate test per
documentation source file (.hpp, .md, etc. files). Source files
with errors can be skipped (--doxygen-skip) or excluded temporary
(--doxygen-xfail).

Usage:
pytest --doxygen doxygen.log --html doc-generation.html test_doc-generation.py
"""

import re
import copy
import pytest
from utils.log import parse


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption('--doxygen', help='Doxygen log path to run tests for')
    parser.addoption(
        '--doxygen-strip',
        default='tmp_docs/',
        help='Path to strip from paths found in doxygen log')
    parser.addoption(
        '--doxygen-xfail',
        action='append',
        default=[],
        help='A file with relative paths to a files with known failures')
    parser.addoption(
        '--doxygen-skip',
        action='append',
        default=[],
        help='A file with relative paths to a files to exclude from validation')
    parser.addoption(
        '--include_omz',
        type=str,
        required=False,
        default='',
        help='Include link check for omz docs')
    parser.addoption(
        '--include_wb',
        type=str,
        required=False,
        default='',
        help='Include link check for workbench docs')
    parser.addoption(
        '--include_pot',
        type=str,
        required=False,
        default='',
        help='Include link check for pot docs')
    parser.addoption(
        '--include_gst',
        type=str,
        required=False,
        default='',
        help='Include link check for gst docs')
    parser.addoption(
        '--include_ovms',
        type=str,
        required=False,
        default='',
        help='Include link check for ovms')


def read_lists(configs):
    """Read lines from files from configs. Return unique items.
    """
    files = set()
    for config_path in configs:
        try:
            with open(config_path, 'r') as config:
                files.update(map(str.strip, config.readlines()))
        except OSError:
            pass
    return list(files)


def pytest_generate_tests(metafunc):
    """ Generate tests depending on command line options
    """
    # read log
    with open(metafunc.config.getoption('doxygen'), 'r') as log:
        all_files = parse(log.read(), metafunc.config.getoption('doxygen_strip'))

    exclude_links = {'open_model_zoo', 'workbench', 'pot',  'gst', 'omz', 'ovms'}
    if metafunc.config.getoption('include_omz'):
        exclude_links.remove('omz')
    if metafunc.config.getoption('include_wb'):
        exclude_links.remove('workbench')
    if metafunc.config.getoption('include_pot'):
        exclude_links.remove('pot')
    if metafunc.config.getoption('include_gst'):
        exclude_links.remove('gst')
    if metafunc.config.getoption('include_ovms'):
        exclude_links.remove('ovms')

    filtered_keys = filter(lambda line: not any([line.startswith(repo) for repo in exclude_links]), all_files)
    files_with_errors = {key: all_files[key] for key in filtered_keys}
    ref_pattern = "unable to resolve reference to '{}"
    for file, errors in copy.deepcopy(files_with_errors).items():
        for error in errors:
            for ex_link in exclude_links:
                if re.match(re.compile(ref_pattern.format(ex_link)), error):
                    files_with_errors[file].remove(error)
            if not len(errors):
                files_with_errors.pop(file)

    # read mute lists
    marks = dict()
    marks.update(
        (name, pytest.mark.xfail)
        for name in read_lists(metafunc.config.getoption('doxygen_xfail')))
    marks.update(
        (name, pytest.mark.skip)
        for name in read_lists(metafunc.config.getoption('doxygen_skip')))

    # generate tests
    if 'doxygen_errors' in metafunc.fixturenames:
        metafunc.parametrize(
            'doxygen_errors', [
                pytest.param(errors, marks=marks[file])
                if file in marks else errors for file, errors in files_with_errors.items()
            ],
            ids=list(files_with_errors.keys()))
