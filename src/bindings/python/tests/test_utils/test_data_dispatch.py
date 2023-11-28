# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

from tests.utils.helpers import generate_relu_compiled_model
from openvino import Type, Shape, Tensor
from openvino.runtime.utils.data_helpers import _data_dispatch
from openvino.runtime.ie_api import OVDict

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
    assert result.get_element_type() == infer_request.inputs[0].get_element_type()
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


@pytest.mark.parametrize("input_data", 
                         {},
                         [],
                         (),
                         OVDict())
def test_container_types(device, input_data, input_shape):
    result, infer_request = _run_dispatcher(device, input_data, False, input_shape)
    from openvino import compile_model
    from openvino.runtime import op, opset13 as opset
    from itertools import cycle
    def create_model(input_names, output_names):
        inputs = [op.Parameter(Type.i32, Shape([1, 1])) for _ in input_names]
        for name, inp in zip(input_names, inputs):
            inp.set_friendly_name(name)

        adds = [
            opset.add(inp, opset.constant(i, Type.i32))
            for i, (inp, _) in enumerate(zip(cycle(inputs), output_names))
        ]

        outputs = []
        for name, add in zip(output_names, adds):
            output = add.output(0)
            output.tensor.set_names({name})
            outputs.append(output)

        return Model(outputs, inputs)


    inputs_1 = ["input_1"]
    outputs_1 = inputs_2 = ["output_1_1", "outputs_1_2"]
    outputs_2 = ["output_2_1", "outputs_2_2"]
    model_1 = create_model(inputs_1, outputs_1)
    model_2 = create_model(inputs_2, outputs_2)
    compiled_1, compiled_2 = compile_model(model_1), compile_model(model_2)
    example_input = np.array([[1]])
    example_output_1 = compiled_1(example_input)
    example_output_2 = compiled_2(example_output_1)
