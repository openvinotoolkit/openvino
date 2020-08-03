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
import numpy as np
import onnx
import pytest

from tests.test_onnx.utils import run_node, xfail_issue_35914, xfail_issue_35915


@pytest.mark.parametrize(
    "onnx_op, numpy_func, data_type",
    [
        pytest.param("And", np.logical_and, np.bool, marks=xfail_issue_35914),
        pytest.param("Or", np.logical_or, np.bool, marks=xfail_issue_35914),
        pytest.param("Xor", np.logical_xor, np.bool, marks=xfail_issue_35914),
        pytest.param("Equal", np.equal, np.int32, marks=xfail_issue_35915),
        pytest.param("Greater", np.greater, np.int32, marks=xfail_issue_35915),
        pytest.param("Less", np.less, np.int32, marks=xfail_issue_35915),
    ],
)
def test_logical(onnx_op, numpy_func, data_type):
    node = onnx.helper.make_node(onnx_op, inputs=["A", "B"], outputs=["C"], broadcast=1)

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]]).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    ng_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(ng_results, [expected_output])

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array(1).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    ng_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(ng_results, [expected_output])


def test_logical_not():
    input_data = np.array([[False, True, True], [False, True, False], [False, False, True]])
    expected_output = np.logical_not(input_data)

    node = onnx.helper.make_node("Not", inputs=["X"], outputs=["Y"])
    ng_results = run_node(node, [input_data])
    assert np.array_equal(ng_results, [expected_output])
