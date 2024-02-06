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
    assert "get_default_context  is not supported by CPU plugin!" in str(cpu_error.value)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_get_default_context_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()
    # TODO: check context.get_params() and bind related properties:
    #       * CONTEXT_TYPE
    #       * OCL_CONTEXT
    #       * OCL_QUEUE


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_device_tensor_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()
    # Test RemoteTensor class:
    tensor = context.create_device_tensor(ov.Type.f32, ov.Shape([1, 2, 3]), {})
    assert isinstance(tensor, ov.RemoteTensor)
    assert "GPU" in tensor.get_device_name()
    # TODO: check tensor.get_params() and bind related properties:
    #       * MEM_HANDLE
    #       * OCL_CONTEXT
    #       * SHARED_MEM_TYPE
    assert tensor.get_shape() == ov.Shape([1, 2, 3])
    assert tensor.get_element_type() == ov.Type.f32
    assert tensor.get_size() == 6
    assert tensor.get_byte_size() == 24
    assert list(tensor.get_strides()) == [24, 12, 4]
    # Resize tensor:
    tensor.set_shape([1,1,1])
    assert tensor.get_shape()
    assert tensor.get_size() == 1
    assert tensor.get_byte_size() ==4
    assert list(tensor.get_strides()) == [4, 4, 4]
    # Test RemoteTensor class NotImplemented methods:
    with pytest.raises(TypeError) as constructor_error:
        _ = ov.RemoteTensor(np.ones((1,2,3)))
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
def test_cl_context():
    # Check if the API was built with OpenCL:
    try:
        from openvino import ClContext
    except ImportError:
        pytest.skip(
            "OpenVINO was built without support for OpenCL."
        )


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_cl_image_2d_tensor():
    # Check if the API was built with OpenCL:
    try:
        from openvino import ClImage2DTensor
    except ImportError:
        pytest.skip(
            "OpenVINO was built without support for OpenCL."
        )


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_va_wrapper():
    # Check if the API was built with libva:
    try:
        from openvino import VADisplayWrapper
    except ImportError:
        pytest.skip(
            "OpenVINO was built without support for libva."
        )
    display = VADisplayWrapper(None)
    assert isinstance(display, VADisplayWrapper)
    with pytest.warns(RuntimeWarning, match="Release of VADisplay was not succesful!"):
        display.release()


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_va_context():
    # Check if the API was built with libva:
    try:
        from openvino import VADisplayWrapper
        from openvino import VAContext
    except ImportError:

        pytest.skip(
            "OpenVINO was built without support for libva."
        )
