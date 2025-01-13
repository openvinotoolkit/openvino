# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging as log
from pathlib import Path

import pytest
from common.layer_test_class import get_params
from common.utils.common_utils import copy_files_by_pattern

def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    setattr(metafunc.cls, 'tflite', metafunc.config.getoption('tflite'))
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.fixture(scope='session', autouse=True)
def rename_tf_fe_libs(request):
    try:
        import openvino.runtime as rt
        # path below is built considering the use of wheels
        openvino_lib_path = Path(rt.__file__).parent.parent / 'libs'
    except ImportError as err:
        raise Exception("Please set PYTHONPATH to OpenVINO Python or install wheel package") from err

    tf_fe_lib_names = ['libopenvino_tensorflow_fe', 'libopenvino_tensorflow_frontend']

    if request.config.getoption('use_legacy_frontend'):
        log.info('Using legacy frontend...')
        copy_files_by_pattern(openvino_lib_path, tf_fe_lib_names[0], tf_fe_lib_names[1])

