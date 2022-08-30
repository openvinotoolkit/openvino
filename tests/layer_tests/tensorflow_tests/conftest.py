# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import sys
import logging as log
from pathlib import Path

import pytest
from common.layer_test_class import get_params

from common.utils.common_utils import rename_ov_lib


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.fixture(scope='session', autouse=True)
def rename_tf_fe_libs(request):
    # code before 'yield' statement is equal to 'set_up' function
    try:
        import openvino.runtime as rt
    except ImportError as err:
        raise Exception("Please set PYTHONPATH to OpenVINO Python") from err
    
    openvino_lib_path = Path(rt.__file__).parent.parent.parent.parent.parent

    if sys.platform == 'win32':
        tf_fe_lib_names = [('openvino_tensorflow_fe.dll', 'openvino_tensorflow_frontend.dll'),
                           ('openvino_tensorflow_fe.lib', 'openvino_tensorflow_frontend.lib'),
                           ('openvino_tensorflow_fe.exp', 'openvino_tensorflow_frontend.exp')]
    else:
        tf_fe_lib_names = [('libopenvino_tensorflow_fe.so', 'libopenvino_tensorflow_frontend.so')]

    if request.config.getoption('use_new_frontend'):
        log.info('Using new frontend...')

        # check if all required files already have new names
        if all([file_pair[1] in os.listdir(openvino_lib_path) for file_pair in tf_fe_lib_names]):
            log.info('TF FE libraries already have new names, no renaming will be done')
        else:
            rename_ov_lib(tf_fe_lib_names, openvino_lib_path)

    # code after 'yield' statement is equal to 'tear_down' function
    yield

    # check if all required files already have old names
    if all([file_pair[0] in os.listdir(openvino_lib_path) for file_pair in tf_fe_lib_names]):
        log.info('TF FE libraries already have old names, no renaming will be done')
    else:
        if sys.platform == 'win32':
            tf_fe_lib_names = [('openvino_tensorflow_frontend.dll', 'openvino_tensorflow_fe.dll'),
                               ('openvino_tensorflow_frontend.lib', 'openvino_tensorflow_fe.lib'),
                               ('openvino_tensorflow_frontend.exp', 'openvino_tensorflow_fe.exp')]
        else:
            tf_fe_lib_names = [('libopenvino_tensorflow_frontend.so', 'libopenvino_tensorflow_fe.so')]
        rename_ov_lib(tf_fe_lib_names, openvino_lib_path)

