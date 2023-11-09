# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import os.path
import shutil
import pytest

from models_hub_common.constants import performance_results_path
from models_hub_common.constants import wget_cache_dir
from models_hub_common.utils import get_params


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


def pytest_sessionstart(session):
    objs_to_remove = []
    if performance_results_path and os.path.exists(performance_results_path):
        objs_to_remove.append(performance_results_path)
    for file_name in os.listdir(wget_cache_dir):
        file_path = os.path.join(wget_cache_dir, file_name)
        objs_to_remove.append(file_path)
    for file_path in objs_to_remove:
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            pass
