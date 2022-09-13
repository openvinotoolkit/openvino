# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging as log
import os
from pathlib import Path

import pytest
from common.layer_test_class import get_params
from common.utils.common_utils import rename_files_by_pattern


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.fixture(scope='session', autouse=True)
def rename_tf_fe_libs(request):
    # code before 'yield' statement is equal to 'set_up' function
    if os.getenv('LD_LIBRARY_PATH'):
        openvino_lib_path = os.getenv('LD_LIBRARY_PATH')
    else:
        try:
            import openvino.runtime as rt
            openvino_lib_path = Path(rt.__file__).parent.parent.parent.parent.parent
        except ImportError as err:
            raise Exception("Please set PYTHONPATH to OpenVINO Python") from err

    tf_fe_lib_names = ['libopenvino_tensorflow_fe', 'libopenvino_tensorflow_frontend']

    # in case of usual test run we should check names of libs and rename back them if applicable
    if not request.config.getoption('use_new_frontend'):
        rename_files_by_pattern(openvino_lib_path, tf_fe_lib_names[1], tf_fe_lib_names[0])

    # in case of new frontend usage we should rename libs
    else:
        log.info('Using new frontend...')
        rename_files_by_pattern(openvino_lib_path, tf_fe_lib_names[0], tf_fe_lib_names[1])

    # code after 'yield' statement is equal to 'tear_down' function
    yield

    # we should rename back names of libs in case of previous renaming
    rename_files_by_pattern(openvino_lib_path, tf_fe_lib_names[1], tf_fe_lib_names[0])
