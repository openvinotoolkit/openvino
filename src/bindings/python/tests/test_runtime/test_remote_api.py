# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import numpy as np

import openvino as ov

from tests.utils.helpers import get_relu_model


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


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_copy_host_to_device_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    host_tensor_ref = ov.Tensor(ov.Type.f32, ov.Shape([1, 2, 3]))

    random_arr = np.random.rand(*host_tensor_ref.shape).astype(np.float32)
    host_tensor_ref.data[:] = random_arr

    # allocate remote tensor with smaller shape and expect proper reallocation
    device_tensor = context.create_tensor(ov.Type.f32, ov.Shape([1, 1, 1]), {})

    # copy to device tensor from host tensor
    host_tensor_ref.copy_to(device_tensor)

    assert host_tensor_ref.get_shape() == device_tensor.get_shape()
    assert host_tensor_ref.get_byte_size() == device_tensor.get_byte_size()

    host_tensor_res = ov.Tensor(ov.Type.f32, ov.Shape([1, 2, 3]))

    # copy from device tensor from host tensor
    host_tensor_res.copy_from(device_tensor)

    assert np.array_equal(host_tensor_res.data, host_tensor_ref.data)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_copy_device_to_host_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    host_tensor_ref = ov.Tensor(ov.Type.f32, ov.Shape([1, 2, 3]))

    random_arr = np.random.rand(*host_tensor_ref.shape).astype(np.float32)
    host_tensor_ref.data[:] = random_arr

    # allocate remote tensor with smaller shape and expect proper reallocation
    device_tensor = context.create_tensor(ov.Type.f32, ov.Shape([1, 1, 1]), {})

    # copy from host tensor to  device tensor
    device_tensor.copy_from(host_tensor_ref)

    assert host_tensor_ref.get_shape() == device_tensor.get_shape()
    assert host_tensor_ref.get_byte_size() == device_tensor.get_byte_size()

    host_tensor_res = ov.Tensor(ov.Type.f32, ov.Shape([1, 2, 3]))

    # copy to host tensor from device tensor
    device_tensor.copy_to(host_tensor_res)

    assert np.array_equal(host_tensor_res.data, host_tensor_ref.data)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_roi_copy_host_to_device_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    host_tensor_ref = ov.Tensor(ov.Type.f32, ov.Shape([4, 4, 4]))

    random_arr = np.random.rand(*host_tensor_ref.shape).astype(np.float32)
    host_tensor_ref.data[:] = random_arr

    begin_roi = ov.runtime.Coordinate([0, 0, 0])
    end_roi = ov.runtime.Coordinate([3, 4, 4])
    roi_host_tensor_ref = ov.Tensor(host_tensor_ref, begin_roi, end_roi)

    device_tensor = context.create_tensor(ov.Type.f32, ov.Shape([4, 4, 4]), {})
    roi_device_tensor = ov.RemoteTensor(device_tensor, begin_roi, end_roi)

    # copy to roi device tensor from roi host tensor
    roi_host_tensor_ref.copy_to(roi_device_tensor)

    assert roi_host_tensor_ref.get_shape() == roi_device_tensor.get_shape()
    assert roi_host_tensor_ref.get_byte_size() == roi_device_tensor.get_byte_size()

    host_tensor_res = ov.Tensor(ov.Type.f32, roi_host_tensor_ref.get_shape())

    # copy from roi device tensor from roi host tensor
    host_tensor_res.copy_from(roi_device_tensor)

    host_tensor_wo_roi = ov.Tensor(ov.Type.f32, roi_host_tensor_ref.get_shape())
    host_tensor_wo_roi.copy_from(roi_host_tensor_ref)

    assert np.array_equal(host_tensor_res.data, host_tensor_wo_roi.data)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_roi_copy_device_to_host_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    assert isinstance(context, ov.RemoteContext)
    assert "GPU" in context.get_device_name()

    host_tensor_ref = ov.Tensor(ov.Type.f32, ov.Shape([4, 4, 4]))

    random_arr = np.random.rand(*host_tensor_ref.shape).astype(np.float32)
    host_tensor_ref.data[:] = random_arr

    begin_roi = ov.runtime.Coordinate([1, 2, 1])
    end_roi = ov.runtime.Coordinate([3, 4, 4])
    roi_host_tensor_ref = ov.Tensor(host_tensor_ref, begin_roi, end_roi)

    device_tensor = context.create_tensor(ov.Type.f32, ov.Shape([4, 4, 4]), {})
    roi_device_tensor = ov.RemoteTensor(device_tensor, begin_roi, end_roi)

    # copy from roi host tensor to roi device tensor
    roi_device_tensor.copy_from(roi_host_tensor_ref)

    assert roi_host_tensor_ref.get_shape() == roi_device_tensor.get_shape()
    assert roi_host_tensor_ref.get_byte_size() == roi_device_tensor.get_byte_size()

    host_tensor_res = ov.Tensor(ov.Type.f32, roi_host_tensor_ref.get_shape())

    # copy to roi host tensor from roi remote tensor
    host_tensor_res.copy_from(roi_device_tensor)

    host_tensor_wo_roi = ov.Tensor(ov.Type.f32, roi_host_tensor_ref.get_shape())
    host_tensor_wo_roi.copy_from(roi_host_tensor_ref)

    assert np.array_equal(host_tensor_res.data, host_tensor_wo_roi.data)
