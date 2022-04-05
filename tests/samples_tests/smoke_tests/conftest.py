"""
 Copyright (C) 2018-2022 Intel Corporation
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
import pytest
import os
import yaml
import sys
import logging as log
from common.samples_common_test_clas import Environment
from common.common_utils import fix_env_conf

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


def pytest_addoption(parser):
    """Specify command-line options for all plugins"""
    parser.addoption("--env_conf", action="store", help="Path to environment configuration file", default='env_conf')
    parser.addoption("--performance", action="store_true", help="Performance run")


@pytest.fixture(scope="session")
def env_conf(request):
    """Fixture function for command-line option."""
    return request.config.getoption('env_conf')


@pytest.fixture(scope="session")
def performance(request):
    """Fixture function for command-line option."""
    return request.config.getoption('performance')


def pytest_configure(config):
    # Setting common environment:
    with open(config.getoption('env_conf'), "r") as env_conf:
        try:
            Environment.env = fix_env_conf(yaml.safe_load(env_conf))
            # Check mandatory env variables:
            mandatory_env_varibales = ['out_directory', 'models_path', 'test_data', 'samples_data_zip', 'smoke_tests_path', 'samples_path']
            missing_variables = []
            for variable in mandatory_env_varibales:
                if variable not in Environment.env:
                    missing_variables.append(variable)
            if missing_variables:
                raise EnvironmentError("Missing env variables in env_config.yml: {}".format(missing_variables))
        except yaml.YAMLError as exc:
            log.error('Cannot work with yml file')
            raise exc

    # Performance stuff:
    Environment.env['performance'] = config.getoption('performance')
    if Environment.env['performance']:
        # Check mandatory perf_result_path env variable
        if 'perf_result_path' not in Environment.env:
            raise EnvironmentError("perf_result_path variable should be set in env_config.yml")
        # Creating name for csv, it should consist of ie and dldt version
        ie_version = os.environ.get('ie_version', 'ie_version')
        dldt_version = os.environ.get('dldt_path', 'dldt_version')
        dldt_version = os.path.basename(os.path.splitext(dldt_version)[0])
        perf_csv_name = '{}_{}.csv'.format(ie_version, dldt_version)
        # Check if this csv has already exist - create new one with extension _1:
        perf_path = os.path.join(Environment.env['perf_result_path'], perf_csv_name)
        if os.path.isfile(perf_path):
            log.info("Resulting cvs has already exist {}. Created new csv with extension _1".format(perf_path))
            while os.path.isfile(perf_path):
                perf_path = perf_path.replace('.csv', '_1.csv')
        log.info('Creating file for storing performance results: {}'.format(perf_path))
        try:
            with open(perf_path, 'w', newline='') as f:
                pass
            # Setting global variable to use in test ('write_csv' function)
            Environment.env['perf_csv_name'] = perf_path
        except IOError as e:
            log.error('Cannot open file for storing perf result: {}'.format(perf_path))
            raise e


def pytest_make_parametrize_id(config, val):
    return repr(val)
