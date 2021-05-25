import inspect
import os
import shutil

import pytest
from common import constants
from common.layer_utils import union
from common.logger import init_logger


def generate_tests(metafunc, test_generator, **scope_for_marker):
    def generator_attributes(marker_name, scope_for_marker):
        if marker_name in scope_for_marker:
            return scope_for_marker[marker_name]
        else:
            return None

    test_gen_attrs_names = list(inspect.signature(test_generator).parameters)
    params = []
    for marker in metafunc.definition.own_markers:
        _params = []
        #   There is a need to set test cases that are required to run for the specified mark
        if marker.name == "precommit":
            test_gen_attrs = generator_attributes(marker.name, scope_for_marker)
            _params = test_generator(**test_gen_attrs) if test_gen_attrs is not None else test_generator()
        elif marker.name == "nightly":
            _params = test_generator()
        #   Find union between scopes for few marks
        params = union(params, _params)
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


def pytest_make_parametrize_id(config, val, argname):
    return " {0}:{1} ".format(argname, val)


@pytest.fixture(scope="class", autouse=True)
def setup_teardown_test_class():
    init_logger('DEBUG')
    if not os.path.exists(constants.mxnet_models_path):
        os.makedirs(constants.mxnet_models_path)
    if not os.path.exists(constants.ir_path):
        os.makedirs(constants.ir_path)

    yield

    shutil.rmtree(constants.mxnet_models_path, True)
    shutil.rmtree(constants.ir_path, True)
