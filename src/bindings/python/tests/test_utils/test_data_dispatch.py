# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np

from tests.conftest import model_path
from tests.test_utils.test_utils import get_relu_model, generate_image, generate_model_and_image, generate_relu_compiled_model
from openvino.runtime import Model, ConstOutput, Type, Shape, Core, Tensor
from openvino.runtime.utils.data_helpers import _data_dispatch

is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)


def _get_value(value):
    return value.data if isinstance(value, Tensor) else value


def _run_dispatcher(device, input_data, input_shape, is_shared):
    compiled_model = generate_relu_compiled_model(device, input_shape)
    infer_request = compiled_model.create_infer_request()
    result = _data_dispatch(infer_request, input_data, is_shared)

    return result, infer_request


@pytest.mark.parametrize("data_type", [np.float_, np.int_, int, float])
@pytest.mark.parametrize("input_shape", [[], [1]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_scalars_dispatcher(device, data_type, input_shape, is_shared):
    test_data = data_type(2)
    expected = Tensor(np.ndarray([], data_type, np.array(test_data)))

    result, _ = _run_dispatcher(device, test_data, input_shape, is_shared)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape([])
    assert result.get_element_type() == Type(data_type)
    assert result.data == expected.data


@pytest.mark.parametrize("input_shape", [[1], [2, 2]])
@pytest.mark.parametrize("is_shared", [True, False])
def test_tensor_dispatcher(device, input_shape, is_shared):
    array = np.ones(input_shape)

    test_data = Tensor(array, is_shared)

    result, _ = _run_dispatcher(device, test_data, input_shape, is_shared)

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

    result, _ = _run_dispatcher(device, test_data, input_shape, True)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == Type(test_data.dtype)
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1], [2, 2]])
def test_ndarray_shared_dispatcher_casting(device, input_shape):
    test_data = np.ones(input_shape)

    result, infer_request = _run_dispatcher(device, test_data, input_shape, True)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == infer_request.inputs[0].get_element_type()
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1, 2, 3], [2, 2]])
def test_ndarray_shared_dispatcher_misalign(device, input_shape):
    test_data = np.asfortranarray(np.ones(input_shape).astype(np.float32))

    result, _ = _run_dispatcher(device, test_data, input_shape, True)

    assert isinstance(result, Tensor)
    assert result.get_shape() == Shape(test_data.shape)
    assert result.get_element_type() == Type(test_data.dtype)
    assert np.array_equal(result.data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(result.data, test_data)


@pytest.mark.parametrize("input_shape", [[1, 2, 3], [2, 2]])
def test_ndarray_copied_dispatcher(device, input_shape):
    test_data = np.ones(input_shape)

    result, infer_request = _run_dispatcher(device, test_data, input_shape, False)

    assert result == {}
    assert np.array_equal(infer_request.inputs[0].data, test_data)

    test_data[0] = 2.0

    assert not np.array_equal(infer_request.inputs[0].data, test_data)
