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

import pytest
from utils.log import LogParser


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption('--doxygen', help='Doxygen log path to run tests for')
    parser.addoption('--sphinx', help='Sphinx log path to run tests for')
    parser.addoption(
        '--doxygen-strip',
        default='tmp_docs/',
        help='Path to strip from paths found in doxygen log')
    parser.addoption(
        '--sphinx-strip',
        default='tmp_docs/',
        help='Path to strip from paths found in sphinx log')
    parser.addoption(
        '--suppress-warnings',
        action='append',
        default=[],
        help='A list of warning patterns to suppress')
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
        action="store_true",
        default=False,
        help='Include link check for omz docs')
    parser.addoption(
        '--include_wb',
        action="store_true",
        default=False,
        help='Include link check for workbench docs')
    parser.addoption(
        '--include_pot',
        action="store_true",
        default=False,
        help='Include link check for pot docs')
    parser.addoption(
        '--include_gst',
        action="store_true",
        default=False,
        help='Include link check for gst docs')
    parser.addoption(
        '--include_ovms',
        action="store_true",
        default=False,
        help='Include link check for ovms')
    parser.addoption(
        '--include_ote',
        action="store_true",
        default=False,
        help='Include link check for ote')


def read_lists(configs):
    """Read lines from files from configs. Return unique items.
    """
    files = set()
    for config_path in configs:
        try:
            with open(config_path, 'r', encoding='utf-8') as config:
                files.update(map(str.strip, config.readlines()))
        except OSError:
            pass
    return list(files)


def pytest_generate_tests(metafunc):
    """ Generate tests depending on command line options
    """
    exclude_links = {'open_model_zoo', 'workbench', 'pot', 'gst', 'omz', 'ovms', 'ote'}
    if metafunc.config.getoption('include_omz'):
        exclude_links.remove('open_model_zoo')
        exclude_links.remove('omz')
    if metafunc.config.getoption('include_wb'):
        exclude_links.remove('workbench')
    if metafunc.config.getoption('include_pot'):
        exclude_links.remove('pot')
    if metafunc.config.getoption('include_gst'):
        exclude_links.remove('gst')
    if metafunc.config.getoption('include_ovms'):
        exclude_links.remove('ovms')
    if metafunc.config.getoption('include_ote'):
        exclude_links.remove('ote')

    # warnings to ignore
    suppress_warnings = read_lists(metafunc.config.getoption('suppress_warnings'))
    for link in exclude_links:
        doxy_ref_pattern = "unable to resolve reference to '{}".format(link)
        sphinx_ref_pattern = "toctree contains reference to nonexisting document '{}".format(link)
        sphinx_ref_pattern2 = "unknown document: {}".format(link)
        suppress_warnings.append(doxy_ref_pattern)
        suppress_warnings.append(sphinx_ref_pattern)
        suppress_warnings.append(sphinx_ref_pattern2)

    xfail_list = [xfail.lower() for xfail in read_lists(metafunc.config.getoption('doxygen_xfail'))]

    # read doxygen log
    doxy_parser = LogParser(metafunc.config.getoption('doxygen'),
                            strip=metafunc.config.getoption('doxygen_strip'),
                            xfail_list=xfail_list,
                            suppress_warnings=suppress_warnings)
    doxy_parser.parse()
    doxygen_warnings = doxy_parser.filter()

    # read sphinx log
    sphinx_parser = LogParser(metafunc.config.getoption('sphinx'),
                              strip=metafunc.config.getoption('sphinx_strip'),
                              xfail_list=xfail_list,
                              suppress_warnings=suppress_warnings
                              )
    sphinx_parser.parse()
    sphinx_warnings = sphinx_parser.filter()

    all_warnings = dict()
    all_warnings.update(doxygen_warnings)
    all_warnings.update(sphinx_warnings)

    filtered_keys = filter(lambda line: not any([line.startswith(repo) for repo in exclude_links]), all_warnings)
    files_with_errors = {key: all_warnings[key] for key in filtered_keys}

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
