# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ctypes
import mmap
import os
import pytest

import numpy as np

import openvino as ov

from tests.utils.helpers import get_relu_model


# The GPU plugin accepts host memory passed as CPU_VA only at an address aligned to GPU_CACHELINE_SIZE.
# mmap returns memory aligned to a 4 KB page, which is a multiple of any GPU cacheline size.
def page_aligned_array(shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.frombuffer(mmap.mmap(-1, nbytes), dtype=dtype).reshape(shape)


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
def test_get_params_enum_values_gpu():
    core = ov.Core()
    context = core.get_default_context("GPU")
    tensor = context.create_tensor(ov.Type.f32, ov.Shape([1]), {})

    assert context.get_params()["CONTEXT_TYPE"].value == "OCL"
    assert tensor.get_params()["SHARED_MEM_TYPE"].value == "OCL_BUFFER"


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
@pytest.mark.parametrize("to_address", [int, np.uint64, np.int64, ctypes.c_void_p, ov.OVAny])
def test_create_tensor_from_host_address_gpu(device, to_address):
    core = ov.Core()
    context = core.get_default_context(device)
    # The GPU plugin also requires CPU_VA_SIZE to be a multiple of GPU_CACHELINE_SIZE.
    cacheline_size = core.get_property(device, "GPU_CACHELINE_SIZE")
    expected = np.arange(cacheline_size // 4, dtype=np.float32)
    buffer = page_aligned_array(expected.shape, np.float32)
    buffer[:] = expected

    tensor = context.create_tensor(
        ov.Type.f32,
        ov.Shape(expected.shape),
        {"SHARED_MEM_TYPE": "CPU_VA", "CPU_VA": to_address(buffer.ctypes.data), "CPU_VA_SIZE": buffer.nbytes},
    )
    result = ov.Tensor(ov.Type.f32, ov.Shape(expected.shape))
    tensor.copy_to(result)

    assert np.array_equal(result.data, expected)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_tensor_from_device_memory_handle_gpu(device):
    core = ov.Core()
    context = core.get_default_context(device)
    owner = context.create_tensor(ov.Type.f32, ov.Shape([16]), {})
    mem_handle = owner.get_params()["MEM_HANDLE"].value

    alias = context.create_tensor(
        ov.Type.f32,
        ov.Shape([16]),
        {"SHARED_MEM_TYPE": "OCL_BUFFER", "MEM_HANDLE": mem_handle},
    )
    expected = np.arange(16, dtype=np.float32)
    owner.copy_from(ov.Tensor(expected))
    result = ov.Tensor(ov.Type.f32, ov.Shape([16]))
    alias.copy_to(result)

    assert np.array_equal(result.data, expected)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
@pytest.mark.parametrize("address", [True, np.True_, -1, 2**64, 1.0, "4096", ov.OVAny("4096")])
def test_create_tensor_rejects_invalid_address_gpu(device, address):
    core = ov.Core()
    context = core.get_default_context(device)

    with pytest.raises(RuntimeError) as address_error:
        context.create_tensor(
            ov.Type.f32,
            ov.Shape([16]),
            {"SHARED_MEM_TYPE": "CPU_VA", "CPU_VA": address, "CPU_VA_SIZE": 64},
        )
    assert "The value of CPU_VA property should be an int address" in str(address_error.value)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_context_with_none_handle_gpu(device):
    core = ov.Core()
    with pytest.raises(RuntimeError) as context_error:
        core.create_context(device, {"CONTEXT_TYPE": "OCL", "OCL_CONTEXT": None})
    assert "user handle is nullptr!" in str(context_error.value)


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_create_context_accepts_ovany_handles_from_get_params_gpu(device):
    core = ov.Core()
    default_params = core.get_default_context(device).get_params()

    context = core.create_context(device, default_params)

    assert context.get_params()["OCL_CONTEXT"].value == default_params["OCL_CONTEXT"].value


@pytest.mark.skipif(
    "GPU" not in os.environ.get("TEST_DEVICE", ""),
    reason="Test can be only performed on GPU device!",
)
def test_infer_in_shared_context_with_user_memory_gpu(device):
    core = ov.Core()
    default_params = core.get_default_context(device).get_params()
    ocl_context = default_params["OCL_CONTEXT"].value
    ocl_queue = default_params["OCL_QUEUE"].value
    assert isinstance(ocl_context, int)
    assert isinstance(ocl_queue, int)

    context = core.create_context(device, {"CONTEXT_TYPE": "OCL", "OCL_CONTEXT": ocl_context, "OCL_QUEUE": ocl_queue})
    assert context.get_params()["OCL_CONTEXT"].value == ocl_context

    model = get_relu_model()
    compiled = core.compile_model(model, context, {"INFERENCE_PRECISION_HINT": "f32"})
    shape = list(model.input().shape)
    input_data = page_aligned_array(shape, np.float32)
    input_data[:] = np.random.uniform(-1, 1, shape)
    output_data = page_aligned_array(shape, np.float32)

    def wrap(array):
        return context.create_tensor(
            ov.Type.f32,
            ov.Shape(shape),
            {"SHARED_MEM_TYPE": "CPU_VA", "CPU_VA": array.ctypes.data, "CPU_VA_SIZE": array.nbytes},
        )

    request = compiled.create_infer_request()
    request.set_tensor(compiled.input().get_any_name(), wrap(input_data))
    request.set_tensor(compiled.output().get_any_name(), wrap(output_data))
    request.start_async()
    request.wait()

    assert np.array_equal(output_data, np.maximum(input_data, 0))


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

    begin_roi = ov.Coordinate([0, 0, 0])
    end_roi = ov.Coordinate([3, 4, 4])
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

    begin_roi = ov.Coordinate([1, 2, 1])
    end_roi = ov.Coordinate([3, 4, 4])
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
