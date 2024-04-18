# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

from models_hub_common.utils import get_params
from models_hub_common.constants import hf_cache_dir, no_clean_cache_dir
from models_hub_common.utils import cleanup_dir


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")

def pytest_sessionfinish(session, exitstatus):
    # remove all downloaded files from cache
    if not no_clean_cache_dir:
        cleanup_dir(hf_cache_dir)
