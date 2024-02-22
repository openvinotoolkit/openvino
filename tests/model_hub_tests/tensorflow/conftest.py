# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from models_hub_common.utils import get_params


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")

def pytest_addoption(parser):
    """Specify command-line options"""
    parser.addoption(
        "--scope",
        default="all",
        action="store",
        help="Models scope for testing. Select one of [all, tf_hub, hf, others]. Default: all.")

@pytest.fixture(scope="function")
def models_scope(request):
    """Fixture function for command-line option."""
    return request.config.getoption('scope')