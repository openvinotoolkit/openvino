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
    setattr(metafunc.cls, 'tflite', metafunc.config.getoption('tflite'))
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")
