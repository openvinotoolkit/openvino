"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path

import pytest

test_root = Path(__file__).parent
project_root = test_root.parent


def pytest_addoption(parser):
    parser.addoption(
        "--caffe_logging", action="store_true", default=False, help="Enable Google log"
    )


def pytest_configure(config):
    if not config.getoption('caffe_logging'):
        os.environ['GLOG_minloglevel'] = '2'


@pytest.fixture
def data_dir():
    return project_root / 'data' / 'test_data'


@pytest.fixture
def models_dir():
    return project_root / 'data' / 'test_models'


@pytest.fixture
def mock_path_exists(mocker):
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('pathlib.Path.is_dir', return_value=True)
    mocker.patch('pathlib.Path.is_file', return_value=True)
    mocker.patch('os.path.exists', return_value=True)
