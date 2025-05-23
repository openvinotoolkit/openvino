# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

from pytorch_layer_test_class import get_params


def pytest_generate_tests(metafunc):
    test_gen_attrs_names = list(inspect.signature(get_params).parameters)
    params = get_params()
    metafunc.parametrize(test_gen_attrs_names, params, scope="function")
