# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import numpy as np

import openvino as ov
import openvino.runtime.opset13 as ops

from tests.utils.helpers import generate_image, get_relu_model, generate_model_with_memory


@pytest.mark.skipif(
    "CPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on CPU device!",
)
def test_get_default_context_cpu():
    core = ov.Core()
    with pytest.raises(RuntimeError) as cpu_error:
        _ = core.get_default_context("CPU")
    possible_errors = ["is not supported by CPU plugin!", "Not Implemented"]
    assert any(error in str(cpu_error.value) for error in possible_errors)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_get_default_context_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    context_params = context.get_params()

    assert isinstance(context_params, dict)
    assert list(context_params.keys()) == ["CONTEXT_TYPE", "OCL_CONTEXT", "OCL_QUEUE"]


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_host_tensor_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    tensor = context.create_host_tensor(ov.Type.f32, ov.Shape([1, 2, 3]))

    assert isinstance(tensor, ov.Tensor)
    assert not isinstance(tensor, ov.RemoteTensor)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_device_tensor_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    tensor = context.create_tensor(ov.Type.f32, ov.Shape([1, 2, 3]), {})
    tensor_params = tensor.get_params()

    assert isinstance(tensor_params, dict)
    assert list(tensor_params.keys()) == ["MEM_HANDLE", "OCL_CONTEXT", "SHARED_MEM_TYPE"]

    assert isinstance(tensor, ov.Tensor)
    assert isinstance(tensor, ov.RemoteTensor)
    assert "GPU" in tensor.get_device_name()
    assert tensor.get_shape() == ov.Shape([1, 2, 3])
    assert tensor.get_element_type() == ov.Type.f32
    assert tensor.get_size() == 6
    assert tensor.get_byte_size() == 24
    assert list(tensor.get_strides()) == [24, 12, 4]

    tensor.set_shape([1, 1, 1])
    assert tensor.get_shape()
    assert tensor.get_size() == 1
    assert tensor.get_byte_size() == 4
    assert list(tensor.get_strides()) == [4, 4, 4]

    with pytest.raises(TypeError) as constructor_error:
        _ = ov.RemoteTensor(np.ones((1, 2, 3)))
    assert "No constructor defined!" in str(constructor_error.value)

    with pytest.raises(RuntimeError) as copy_to_error:
        _ = tensor.copy_to(None)
    assert "This function is not implemented." in str(copy_to_error.value)

    with pytest.raises(RuntimeError) as data_error:
        _ = tensor.data
    assert "This function is not implemented." in str(data_error.value)

    with pytest.raises(RuntimeError) as bytes_data_error:
        _ = tensor.bytes_data
    assert "This function is not implemented." in str(bytes_data_error.value)

    with pytest.raises(RuntimeError) as str_data_error:
        _ = tensor.str_data
    assert "This function is not implemented." in str(str_data_error.value)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_compile_with_context():
    core = ov.Core()
    context = core.get_default_context("GPU")
    model = get_relu_model()
    compiled = core.compile_model(model, context)
    assert isinstance(compiled, ov.CompiledModel)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_va_context():
    core = ov.Core()
    with pytest.raises(RuntimeError) as context_error:
        _ = ov.VAContext(core, None)
    assert "user handle is nullptr!" in str(context_error.value)
