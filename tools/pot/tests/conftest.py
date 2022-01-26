# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from .utils.model_store import ModelStore


@pytest.fixture(scope='session')
def models():
    return ModelStore()


@pytest.fixture
def preset(request):
    return request.config.getoption('--preset')


@pytest.fixture
def algorithm(request):
    return request.config.getoption('--algorithm')


def pytest_addoption(parser):
    parser.addoption('--preset', default='performance')
    parser.addoption('--algorithm', default='DefaultQuantization')
