# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
from pathlib import Path

import pytest

from common import constants


def pytest_make_parametrize_id(config, val, argname):
    return " {0}:{1} ".format(argname, val)


def pytest_collection_modifyitems(items):
    def remove_ignored_attrs(ref_dict, dict_to_upd):
        _dict_to_upd = dict_to_upd.copy()
        for key, value in dict_to_upd.items():
            if key not in ref_dict.keys():
                _dict_to_upd.pop(key)
            elif isinstance(value, dict):
                _dict_to_upd[key] = remove_ignored_attrs(ref_dict[key], value)
        return _dict_to_upd

    for test in items:
        special_marks = [mark for mark in test.own_markers if "special_" in mark.name]
        for mark in special_marks:
            if mark.name == "special_xfail":
                params = test.callspec.params
                # Remove items from params if key of item is not in mark.kwargs["args"].
                # Remaining items will be used to mark test cases that contain them.
                # It is required to specify in mark only valuable parameters
                # (e.g. {"device": "FP16"} will mean that for all test cases with FP16 test will be marked)
                params = remove_ignored_attrs(mark.kwargs["args"], params)
                if mark.kwargs["args"] == params:
                    test.add_marker(pytest.mark.xfail(reason=mark.kwargs["reason"]))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    pytest_html = item.config.pluginmanager.getplugin('html')
    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, 'extra', [])
    if report.when == 'call':
        xfail_reason = getattr(report, 'wasxfail', None)
        if report.skipped and xfail_reason:
            jira_ticket_nums = re.findall(r"\*-\d+", xfail_reason)
            for ticket_num in jira_ticket_nums:
                extra.append(pytest_html.extras.url(ticket_num))
        report.extra = extra


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption(
        "--ir_version",
        default=11,
        action="store",
        help="Version of IR to generate by Model Optimizer")
    parser.addoption(
        "--use_legacy_frontend",
        required=False,
        action="store_true",
        help="Use Model Optimizer with legacy FrontEnd")
    parser.addoption(
        "--tflite",
        required=False,
        action="store_true",
        help="Switch to tflite tests version")


@pytest.fixture(scope="session")
def ir_version(request):
    """Fixture function for command-line option."""
    return request.config.getoption('ir_version')


@pytest.fixture(scope="session")
def use_legacy_frontend(request):
    """Fixture function for command-line option."""
    return request.config.getoption('use_legacy_frontend')


@pytest.fixture(scope="session")
def tflite(request):
    """Fixture function for command-line option."""
    return request.config.getoption('tflite')


@pytest.fixture(scope="function")
def temp_dir(request):
    """Create directory for test purposes."""
    Path(constants.out_path).mkdir(parents=True, exist_ok=True)
    test_name = re.sub(r"[^\w_]", "_", request.node.originalname)
    device = request.node.funcargs["ie_device"].upper()
    temp_dir = tempfile.TemporaryDirectory(dir=constants.out_path, prefix=f"{device}_{test_name}")
    yield str(temp_dir.name)
