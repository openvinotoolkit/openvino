# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from tests_compatibility.runtime import get_runtime
from tests_compatibility.test_ngraph.util import run_op_node


def test_onehot():
    runtime = get_runtime()
    param = ng.parameter([3], dtype=np.int32)
    model = ng.one_hot(param, 3, 1, 0, 0)
    computation = runtime.computation(model, param)

    expected = np.eye(3)[np.array([1, 0, 2])]
    input_data = np.array([1, 0, 2], dtype=np.int32)
    result = computation(input_data)
    assert np.allclose(result, expected)
