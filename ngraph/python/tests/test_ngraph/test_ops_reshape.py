# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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
import pytest

import ngraph as ng
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node, run_op_numeric_data
from tests import xfail_issue_40957


def test_concat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    axis = 0
    expected = np.concatenate((a, b), axis=0)

    runtime = get_runtime()
    parameter_a = ng.parameter(list(a.shape), name="A", dtype=np.float32)
    parameter_b = ng.parameter(list(b.shape), name="B", dtype=np.float32)
    node = ng.concat([parameter_a, parameter_b], axis)
    computation = runtime.computation(node, parameter_a, parameter_b)
    result = computation(a, b)
    assert np.allclose(result, expected)


@xfail_issue_40957
@pytest.mark.parametrize(
    "val_type, value", [(bool, False), (bool, np.empty((2, 2), dtype=bool))]
)
def test_constant_from_bool(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "val_type, value",
    [
        pytest.param(np.float32, np.float32(0.1234), marks=xfail_issue_40957),
        pytest.param(np.float64, np.float64(0.1234), marks=xfail_issue_40957),
        pytest.param(np.int8, np.int8(-63), marks=xfail_issue_40957),
        pytest.param(np.int16, np.int16(-12345), marks=xfail_issue_40957),
        pytest.param(np.int32, np.int32(-123456), marks=xfail_issue_40957),
        pytest.param(np.int64, np.int64(-1234567), marks=xfail_issue_40957),
        pytest.param(np.uint8, np.uint8(63), marks=xfail_issue_40957),
        pytest.param(np.uint16, np.uint16(12345), marks=xfail_issue_40957),
        pytest.param(np.uint32, np.uint32(123456), marks=xfail_issue_40957),
        pytest.param(np.uint64, np.uint64(1234567), marks=xfail_issue_40957),
    ],
)
def test_constant_from_scalar(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "val_type",
    [
        pytest.param(np.float32, marks=xfail_issue_40957),
        pytest.param(np.float64, marks=xfail_issue_40957),
    ],
)
def test_constant_from_float_array(val_type):
    np.random.seed(133391)
    input_data = np.array(-1 + np.random.rand(2, 3, 4) * 2, dtype=val_type)
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)


@xfail_issue_40957
@pytest.mark.parametrize(
    "val_type, range_start, range_end",
    [
        (np.int8, -8, 8),
        (np.int16, -64, 64),
        (np.int32, -1024, 1024),
        (np.int64, -16383, 16383),
        (np.uint8, 0, 8),
        (np.uint16, 0, 64),
        (np.uint32, 0, 1024),
        (np.uint64, 0, 16383),
    ],
)
def test_constant_from_integer_array(val_type, range_start, range_end):
    np.random.seed(133391)
    input_data = np.array(
        np.random.randint(range_start, range_end, size=(2, 2)), dtype=val_type
    )
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)


def test_broadcast_numpy():
    data_shape = [16, 1, 1]
    target_shape_shape = [4]

    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    target_shape_parameter = ng.parameter(
        target_shape_shape, name="Target_shape", dtype=np.int64
    )

    node = ng.broadcast(data_parameter, target_shape_parameter)

    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1


def test_broadcast_bidirectional():
    data_shape = [16, 1, 1]
    target_shape_shape = [4]

    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    target_shape_parameter = ng.parameter(
        target_shape_shape, name="Target_shape", dtype=np.int64
    )

    node = ng.broadcast(data_parameter, target_shape_parameter, "BIDIRECTIONAL")

    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1


def test_gather():
    input_data = np.array(
        [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2], np.float32
    ).reshape((3, 3))
    input_indices = np.array([0, 2], np.int32).reshape(1, 2)
    input_axes = np.array([1], np.int32)

    expected = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float32).reshape(
        (3, 1, 2)
    )

    result = run_op_node([input_data], ng.gather, input_indices, input_axes)
    assert np.allclose(result, expected)


def test_transpose():
    input_tensor = np.arange(3 * 3 * 224 * 224, dtype=np.int32).reshape(
        (3, 3, 224, 224)
    )
    input_order = np.array([0, 2, 3, 1], dtype=np.int32)

    result = run_op_node([input_tensor], ng.transpose, input_order)

    expected = np.transpose(input_tensor, input_order)

    assert np.allclose(result, expected)


@pytest.mark.xfail(
    reason="Tile operation has a form that is not supported. Tile_2 should be converted to TileIE operation."
)
def test_tile():
    input_tensor = np.arange(6, dtype=np.int32).reshape((2, 1, 3))
    repeats = np.array([2, 1], dtype=np.int32)

    result = run_op_node([input_tensor], ng.tile, repeats)

    expected = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]).reshape((2, 2, 3))

    assert np.allclose(result, expected)


@pytest.mark.xfail(
    reason="RuntimeError: Check 'shape_size(get_input_shape(0)) == shape_size(output_shape)'"
)
def test_strided_slice():
    input_tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    begin = np.array([1, 0], dtype=np.int32)
    end = np.array([0, 0], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = np.array([0, 0, 0], dtype=np.int32)
    end_mask = np.array([0, 0, 0], dtype=np.int32)
    new_axis_mask = np.array([0, 1, 0], dtype=np.int32)
    shrink_axis_mask = np.array([1, 0, 0], dtype=np.int32)
    ellipsis_mask = np.array([0, 0, 0], dtype=np.int32)

    result = run_op_node(
        [input_tensor],
        ng.strided_slice,
        begin,
        end,
        strides,
        begin_mask,
        end_mask,
        new_axis_mask,
        shrink_axis_mask,
        ellipsis_mask,
    )

    expected = np.array(
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.float32
    ).reshape((1, 3, 4))

    assert np.allclose(result, expected)


def test_reshape_v1():
    A = np.arange(1200, dtype=np.float32).reshape((2, 5, 5, 24))
    shape = np.array([0, -1, 4], dtype=np.int32)
    special_zero = True

    expected_shape = np.array([2, 150, 4])
    expected = np.reshape(A, expected_shape)
    result = run_op_node([A], ng.reshape, shape, special_zero)

    assert np.allclose(result, expected)


def test_shape_of():
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    result = run_op_node([input_tensor], ng.shape_of)

    assert np.allclose(result, [3, 3])
