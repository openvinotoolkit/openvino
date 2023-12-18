# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from copy import deepcopy
import numpy as np

from tests.utils.helpers import generate_relu_compiled_model

from openvino import Core, Model, Type, Shape, Tensor
import openvino.runtime.opset13 as ops
from openvino.runtime.utils.data_helpers import _data_dispatch

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"


def _get_value(value):
    return value.data if isinstance(value, Tensor) else value


def _run_dispatcher(device, input_data, is_shared, input_shape, input_dtype=np.float32):
    compiled_model = generate_relu_compiled_model(device, input_shape, input_dtype)
    infer_request = compiled_model.create_infer_request()
    result = _data_dispatch(infer_request, input_data, is_shared)

    return result, infer_request


@pytest.mark.parametrize("data_type", [np.float_, np.int_, int, float])
@pytest.mark.parametrize("input_shape", [[], [1]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_scalars_dispatcher_old(device, data_type, input_shape, is_shared):
    test_data = data_type(2)
    expected = Tensor(np.ndarray([], data_type, np.array(test_data)))

    result, _ = _run_dispatcher(device, test_data, is_shared, input_shape)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape([])
    assert result.get_element_type() == Type(np.float32)
    assert result.data == expected.data


@pytest.mark.parametrize(("input_data", "input_dtype"), [
    (np.float_(2), np.float_),
    (np.int_(1), np.int_),
    (int(3), np.int64),
    (1, np.int64),
    (float(7), np.float64),
    (1.0, np.int64),
])
@pytest.mark.parametrize("input_shape", [[], [1]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_scalars_dispatcher_new_0(device, input_data, input_dtype, input_shape, is_shared):
    expected = Tensor(np.array(input_data, dtype=input_dtype))

    result, _ = _run_dispatcher(device, input_data, is_shared, input_shape, input_dtype)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape([])
    assert result.get_element_type() == Type(input_dtype)
    assert result.data == expected.data


@pytest.mark.parametrize(("input_data", "is_shared", "expected"), [
    (np.array(2.0, dtype=np.float32), False, {}),
    (np.array(2.0, dtype=np.float32), True, Tensor(np.array(2.0, dtype=np.float32))),
    (np.array(1, dtype=np.int8), False, {}),
    (np.array(1, dtype=np.int8), True, Tensor(np.array(1, dtype=np.float32))),
])
@pytest.mark.parametrize("input_shape", [[], [1]])
def test_scalars_dispatcher_new_1(device, input_data, is_shared, expected, input_shape):
    result, _ = _run_dispatcher(device, input_data, is_shared, input_shape, np.float32)

    assert isinstance(result, type(expected))
    if isinstance(result, dict):
        assert len(result) == 0
    else:
        assert result.get_shape() == Shape(input_shape)
        assert result.get_element_type() == Type(np.float32)
        assert result.data == expected.data


@pytest.mark.parametrize("input_shape", [[1], [2, 2]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_tensor_dispatcher(device, input_shape, is_shared):
    array = np.ones(input_shape)

    test_data = Tensor(array, is_shared)

    result, _ = _run_dispatcher(device, test_data, is_shared, input_shape)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(input_shape)
    assert result.get_element_type() == Type(array.dtype)
    assert np.array_equal(result.data, array)

    # Change data to check if shared_memory is still applied
    array[0] = 2.0

    assert np.array_equal(array, result.data) if is_shared else not np.array_equal(array, result.data)


@pytest.mark.parametrize("input_shape", [[1], [2, 2]])
def test_ndarray_shared_dispatcher(device, input_shape):
    test_data = np.ones(input_shape).astype(np.float32)

    result, _ = _run_dispatcher(device, test_data, True, input_shape)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == Type(test_data.dtype)
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1], [2, 2]])
def test_ndarray_shared_dispatcher_casting(device, input_shape):
    test_data = np.ones(input_shape)

    result, infer_request = _run_dispatcher(device, test_data, True, input_shape)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == infer_request.input_tensors[0].get_element_type()
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1, 2, 3], [2, 2]])
def test_ndarray_shared_dispatcher_misalign(device, input_shape):
    test_data = np.asfortranarray(np.ones(input_shape).astype(np.float32))

    result, _ = _run_dispatcher(device, test_data, True, input_shape)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == Type(test_data.dtype)
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1, 2, 3], [2, 2]])
def test_ndarray_copied_dispatcher(device, input_shape):
    test_data = np.ones(input_shape)

    result, infer_request = _run_dispatcher(device, test_data, False, input_shape)

    assert result == {}
    assert np.array_equal(infer_request.input_tensors[0].data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(infer_request.input_tensors[0].data, test_data)


@pytest.mark.parametrize(
    ("input_data"),
    [
        np.array(["śćżóąę", "data_dispatcher_test"]),
        np.array(["abcdef", "data_dispatcher_test"]).astype("S"),
    ],
)
@pytest.mark.parametrize("data_type", [Type.string, str, bytes, np.str_, np.bytes_])
@pytest.mark.parametrize("input_shape", [[2], [2, 1]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_string_array_dispatcher(device, input_data, data_type, input_shape, is_shared):
    # Copy data so it won't be overriden by next testcase:
    test_data = np.copy(input_data).reshape(input_shape)

    param = ops.parameter(input_shape, data_type, name="data")
    res = ops.result(param)
    model = Model([res], [param], "test_model")

    core = Core()

    compiled_model = core.compile_model(model, device)

    infer_request = compiled_model.create_infer_request()
    result = _data_dispatch(infer_request, test_data, is_shared)

    if is_shared:
        assert isinstance(result, Tensor)
        assert result.element_type == Type.string
        assert result.shape == Shape(input_shape)
        if test_data.dtype.kind == "U":
            assert np.array_equal(result.bytes_data, np.char.encode(test_data))
            assert np.array_equal(result.str_data, test_data)
        else:
            assert np.array_equal(result.bytes_data, test_data)
            assert np.array_equal(result.str_data, np.char.decode(test_data))
        assert not np.shares_memory(result.bytes_data, test_data)
        assert not np.shares_memory(result.str_data, test_data)
    else:
        assert result == {}
        if test_data.dtype.kind == "U":
            assert np.array_equal(infer_request.input_tensors[0].bytes_data, np.char.encode(test_data))
            assert np.array_equal(infer_request.input_tensors[0].str_data, test_data)
        else:
            assert np.array_equal(infer_request.input_tensors[0].bytes_data, test_data)
            assert np.array_equal(infer_request.input_tensors[0].str_data, np.char.decode(test_data))
        assert not np.shares_memory(infer_request.input_tensors[0].bytes_data, test_data)
        assert not np.shares_memory(infer_request.input_tensors[0].str_data, test_data)
        # Override value to confirm:
        test_data[0] = "different string"
        if test_data.dtype.kind == "U":
            assert not np.array_equal(infer_request.input_tensors[0].bytes_data, np.char.encode(test_data))
            assert not np.array_equal(infer_request.input_tensors[0].str_data, test_data)
        else:
            assert not np.array_equal(infer_request.input_tensors[0].bytes_data, test_data)
            assert not np.array_equal(infer_request.input_tensors[0].str_data, np.char.decode(test_data))


@pytest.mark.parametrize(
    ("input_data", "input_shape"),
    [
        (["śćżóąę", "data_dispatcher_test"], [2]),
        ([b"abcdef", b"data_dispatcher_test"], [2]),
        ([bytes("abc", encoding="utf-8"), bytes("zzzz", encoding="utf-8")], [2]),
        ([["śćżóąę", "data_dispatcher_test"]], [1, 2]),
        ([["śćżóąę"], ["data_dispatcher_test"]], [2, 1]),
    ],
)
@pytest.mark.parametrize("data_type", [Type.string, str, bytes, np.str_, np.bytes_])
@pytest.mark.parametrize("is_shared", [True, False])
def test_string_list_dispatcher(device, input_data, input_shape, data_type, is_shared):
    # Copy data so it won't be overriden by next testcase:
    test_data = deepcopy(input_data)
    param = ops.parameter(input_shape, data_type, name="data")
    res = ops.result(param)
    model = Model([res], [param], "test_model")

    core = Core()

    compiled_model = core.compile_model(model, device)

    infer_request = compiled_model.create_infer_request()
    result_dict = _data_dispatch(infer_request, test_data, is_shared)

    # Dispatcher will always return new Tensors from any lists.
    # For copied approach it will be based of the list and ov.Tensor class
    # is responsible for copying list over to C++ memory.
    result = result_dict[0]
    assert isinstance(result, Tensor)
    assert result.element_type == Type.string
    assert result.shape == Shape(input_shape)

    # Convert input_data into numpy array to test properties
    test_data_np = np.array(input_data).reshape(input_shape)

    if test_data_np.dtype.kind == "U":
        assert np.array_equal(result.bytes_data, np.char.encode(test_data_np))
        assert np.array_equal(result.str_data, test_data_np)
    else:
        assert np.array_equal(result.bytes_data, test_data_np)
        assert np.array_equal(result.str_data, np.char.decode(test_data_np))


@pytest.mark.parametrize(
    ("input_data"),
    [
        "śćżóąę",
        "test dispatcher",
        bytes("zzzz", encoding="utf-8"),
        b"aaaaaaa",
        "😁😁",
    ],
)
@pytest.mark.parametrize("data_type", [Type.string, str, bytes, np.str_, np.bytes_])
@pytest.mark.parametrize("is_shared", [True, False])
def test_string_scalar_dispatcher(device, input_data, data_type, is_shared):
    test_data = input_data

    param = ops.parameter([1], data_type, name="data")
    res = ops.result(param)
    model = Model([res], [param], "test_model")

    core = Core()

    compiled_model = core.compile_model(model, device)

    infer_request = compiled_model.create_infer_request()
    result = _data_dispatch(infer_request, test_data, is_shared)

    # Result will always be a Tensor:
    assert isinstance(result, Tensor)
    assert result.element_type == Type.string
    assert result.shape == Shape([])
    if isinstance(test_data, str):
        assert np.array_equal(result.bytes_data, np.char.encode(test_data))
        assert np.array_equal(result.str_data, test_data)
    else:
        assert np.array_equal(result.bytes_data, test_data)
        assert np.array_equal(result.str_data, np.char.decode(test_data))
    assert not np.shares_memory(result.bytes_data, test_data)
    assert not np.shares_memory(result.str_data, test_data)
