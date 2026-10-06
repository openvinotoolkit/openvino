# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

from models_hub_common.utils import get_params
from models_hub_common import env_select


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")


# Re-export of models_hub_common.env_select's hook. Not registered via `pytest_plugins` (a
# non-rootdir conftest raises "Plugins registered by the pytest_plugins variable are only allowed
# in the root conftest.py file" -- there is no rootdir conftest.py here). Wrapping the function is
# the standard equivalent within one plugin. See env_select.py for what/how/why.
def pytest_collection_modifyitems(config, items):
    env_select.pytest_collection_modifyitems(config, items)
