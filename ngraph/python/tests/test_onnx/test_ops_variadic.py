# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from functools import reduce

import numpy as np
import onnx
import pytest

from tests.test_onnx.utils import run_node
from tests import xfail_issue_35926


@xfail_issue_35926
@pytest.mark.parametrize("onnx_op,numpy_func", [("Sum", np.add), ("Min", np.minimum), ("Max", np.maximum)])
def test_variadic(onnx_op, numpy_func):
    data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    node = onnx.helper.make_node(onnx_op, inputs=["data_0", "data_1", "data_2"], outputs=["y"])
    expected_output = reduce(numpy_func, data)

    ng_results = run_node(node, data)
    assert np.array_equal(ng_results, [expected_output])


@xfail_issue_35926
def test_mean():
    data = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    node = onnx.helper.make_node("Mean", inputs=["data_0", "data_1", "data_2"], outputs=["y"])
    expected_output = reduce(np.add, data) / len(data)

    ng_results = run_node(node, data)
    assert np.array_equal(ng_results, [expected_output])
