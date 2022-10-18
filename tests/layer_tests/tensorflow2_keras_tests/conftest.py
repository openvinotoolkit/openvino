import inspect
import logging as log
import os
from pathlib import Path

import pytest
from common.layer_test_class import get_params
from common.logger import *
from common.utils.common_utils import copy_files_by_pattern


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


@pytest.fixture(scope='session', autouse=True)
def rename_tf_fe_libs(request):
    if os.getenv('OV_FRONTEND_PATH'):
        # use this env variable to define path to your specific libs
        openvino_lib_path = Path(os.getenv('OV_FRONTEND_PATH'))
    else:
        try:
            import openvino.runtime as rt
            # path below is built considering the use of wheels
            openvino_lib_path = Path(rt.__file__).parent.parent / 'libs'
        except ImportError as err:
            raise Exception("Please set PYTHONPATH to OpenVINO Python or install wheel package "
                            "or use OV_FRONTEND_PATH env variable") from err

    tf_fe_lib_names = ['libopenvino_tensorflow_fe', 'libopenvino_tensorflow_frontend']

    if request.config.getoption('use_new_frontend'):
        log.info('Using new frontend...')
        copy_files_by_pattern(openvino_lib_path, tf_fe_lib_names[0], tf_fe_lib_names[1])
