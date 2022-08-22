# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import sys
from pathlib import Path

import pytest
from common.layer_test_class import get_params

from tests.layer_tests.common.utils.common_utils import rename_ov_lib


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.fixture(scope='session', autouse=True)
def rename_tf_fe_libs(request):
    try:
        import openvino.runtime as rt
    except ImportError as err:
        raise Exception("Please set PYTHONPATH to OpenVINO Python") from err
    
    openvino_lib_path = Path(rt.__file__).parent.parent.parent.parent.parent

    if sys.platform == 'win32':
        tf_fe_lib_names = ['openvino_tensorflow_fe.dll', 'openvino_tensorflow_frontend.dll']
    else:
        tf_fe_lib_names = ['libopenvino_tensorflow_fe.so', 'libopenvino_tensorflow_frontend.so']

    if request.config.getoption('use_new_frontend'):
        print('Using new frontend...')

        if tf_fe_lib_names[1] in os.listdir(openvino_lib_path):
            print('TF FE library is already has new name, no renaming will be done')
        else:
            rename_ov_lib(tf_fe_lib_names[0], tf_fe_lib_names[1], openvino_lib_path)

        yield

        if tf_fe_lib_names[0] in os.listdir(openvino_lib_path):
            print('TF FE library is already has old name, no renaming will be done')
        else:
            rename_ov_lib(tf_fe_lib_names[1], tf_fe_lib_names[0], openvino_lib_path)
    else:
        if tf_fe_lib_names[1] in os.listdir(openvino_lib_path):
            rename_ov_lib(tf_fe_lib_names[1], tf_fe_lib_names[0], openvino_lib_path)
