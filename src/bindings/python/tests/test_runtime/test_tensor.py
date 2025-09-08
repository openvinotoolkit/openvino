# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy, copy
import os
import subprocess
import sys

import numpy as np

import openvino as ov
import openvino.opset13 as ops
from openvino.helpers import pack_data, unpack_data

import pytest

from tests.utils.helpers import generate_image, generate_relu_compiled_model


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.bf16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
        (ov.Type.u1, np.uint8),
        (ov.Type.u4, np.uint8),
        (ov.Type.i4, np.int8),
    ],
)
def test_init_with_ov_type(ov_type, numpy_dtype):
    ov_tensors = []
    ov_tensors.append(ov.Tensor(type=ov_type, shape=ov.Shape([1, 3, 32, 32])))
    ov_tensors.append(ov.Tensor(type=ov_type, shape=[1, 3, 32, 32]))
    assert np.all(list(ov_tensor.shape) == [1, 3, 32, 32] for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.element_type == ov_type for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.data.dtype == numpy_dtype for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.data.shape == (1, 3, 32, 32) for ov_tensor in ov_tensors)


def test_subprocess():
    args = [sys.executable, os.path.join(os.path.dirname(__file__), "subprocess_test_tensor.py")]

    status = subprocess.run(args, env=os.environ)
    assert not status.returncode


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_init_with_numpy_dtype(ov_type, numpy_dtype):
    shape = (1, 3, 127, 127)
    ov_shape = ov.Shape(shape)
    ov_tensors = []
    ov_tensors.append(ov.Tensor(type=numpy_dtype, shape=shape))
    ov_tensors.append(ov.Tensor(type=np.dtype(numpy_dtype), shape=shape))
    ov_tensors.append(ov.Tensor(type=np.dtype(numpy_dtype), shape=np.array(shape)))
    ov_tensors.append(ov.Tensor(type=numpy_dtype, shape=ov_shape))
    ov_tensors.append(ov.Tensor(type=np.dtype(numpy_dtype), shape=ov_shape))
    assert np.all(tuple(ov_tensor.shape) == shape for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.element_type == ov_type for ov_tensor in ov_tensors)
    assert np.all(isinstance(ov_tensor.data, np.ndarray) for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.data.dtype == numpy_dtype for ov_tensor in ov_tensors)
    assert np.all(ov_tensor.data.shape == shape for ov_tensor in ov_tensors)


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_init_with_numpy_shared_memory(ov_type, numpy_dtype):
    arr = generate_image().astype(numpy_dtype)
    shape = arr.shape
    arr = np.ascontiguousarray(arr)
    ov_tensor = ov.Tensor(array=arr, shared_memory=True)
    assert tuple(ov_tensor.shape) == shape
    assert ov_tensor.element_type == ov_type
    assert isinstance(ov_tensor.data, np.ndarray)
    assert ov_tensor.data.dtype == numpy_dtype
    assert ov_tensor.data.shape == shape
    assert np.shares_memory(arr, ov_tensor.data)
    assert np.array_equal(ov_tensor.data, arr)
    assert ov_tensor.size == arr.size
    assert ov_tensor.byte_size == arr.nbytes
    assert tuple(ov_tensor.strides) == arr.strides

    assert tuple(ov_tensor.get_shape()) == shape
    assert ov_tensor.get_element_type() == ov_type
    assert ov_tensor.get_size() == arr.size
    assert ov_tensor.get_byte_size() == arr.nbytes
    assert tuple(ov_tensor.get_strides()) == arr.strides


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_init_with_numpy_copy_memory(ov_type, numpy_dtype):
    arr = generate_image().astype(numpy_dtype)
    shape = arr.shape
    ov_tensor = ov.Tensor(array=arr, shared_memory=False)
    assert tuple(ov_tensor.shape) == shape
    assert ov_tensor.element_type == ov_type
    assert isinstance(ov_tensor.data, np.ndarray)
    assert ov_tensor.data.dtype == numpy_dtype
    assert ov_tensor.data.shape == shape
    assert not (np.shares_memory(arr, ov_tensor.data))
    assert np.array_equal(ov_tensor.data, arr)
    assert ov_tensor.size == arr.size
    assert ov_tensor.byte_size == arr.nbytes


def test_init_with_node_output_port():
    def get_tensor():
        param1 = ops.parameter(ov.Shape([1, 3, 2, 2]), dtype=np.float64)
        param2 = ops.parameter(ov.Shape([1, 3, 32, 32]), dtype=np.float64)
        param3 = ops.parameter(ov.PartialShape.dynamic(), dtype=np.float64)
        ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float64)
        assert sys.getrefcount(ones_arr) == 2
        tensor1 = ov.Tensor(param1.output(0))
        tensor2 = ov.Tensor(param2.output(0), ones_arr)
        assert sys.getrefcount(ones_arr) == 3
        tensor3 = ov.Tensor(param3.output(0))
        tensor4 = ov.Tensor(param3.output(0), ones_arr)
        assert tensor1.shape == param1.shape
        assert tensor1.element_type == param1.get_element_type()
        assert tensor2.shape == param2.shape
        assert tensor2.element_type == param2.get_element_type()
        assert tensor3.shape == ov.Shape([0])
        assert tensor3.element_type == param3.get_element_type()
        assert tensor4.shape == ov.Shape([0])
        assert tensor4.element_type == param3.get_element_type()
        ones_arr[0][0][0][0:2] = 0

        del ones_arr
        return tensor2

    shared_tensor = get_tensor()
    assert np.allclose(shared_tensor.data[0][0][0][0:3], [0, 0, 1])


def test_init_with_node_constoutput_port(device):
    def get_tensor():
        compiled_model = generate_relu_compiled_model(device)
        output = compiled_model.output(0)
        ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
        assert sys.getrefcount(ones_arr) == 2

        tensor1 = ov.Tensor(output)
        tensor2 = ov.Tensor(output, ones_arr)
        assert sys.getrefcount(ones_arr) == 3

        output_node = output.get_node()
        assert tensor1.shape == output_node.shape
        assert tensor1.element_type == output_node.get_element_type()
        assert tensor2.shape == output_node.shape
        assert tensor2.element_type == output_node.get_element_type()
        assert np.array_equal(tensor2.data, ones_arr)

        ones_arr[0][0][0][0:2] = 0

        del ones_arr
        return tensor2

    tensor = get_tensor()
    assert np.allclose(tensor.data[0][0][0][0:3], [0, 0, 1])


def test_init_with_output_port_different_shapes():
    param1 = ops.parameter(ov.Shape([2]), dtype=np.float32)
    param2 = ops.parameter(ov.Shape([5]), dtype=np.float32)

    ones_arr = np.ones(shape=(2, 2), dtype=np.float32)
    with pytest.warns(RuntimeWarning):
        ov.Tensor(param1.output(0), ones_arr)

    with pytest.raises(RuntimeError) as e:
        ov.Tensor(param2.output(0), ones_arr)
    assert "Shape of the port exceeds shape of the array." in str(e.value)


def test_init_with_output_port_different_types():
    param1 = ops.parameter(ov.Shape([2]), dtype=np.int16)
    ones_arr = np.ones(shape=(2, 2), dtype=np.int8)
    with pytest.warns(RuntimeWarning):
        tensor = ov.Tensor(param1.output(0), ones_arr)
    assert not np.array_equal(tensor.data, ones_arr)


def test_init_with_roi_tensor():
    array = np.random.normal(size=[1, 3, 48, 48])
    ov_tensor1 = ov.Tensor(array)
    ov_tensor2 = ov.Tensor(ov_tensor1, [0, 0, 24, 24], [1, 3, 48, 48])
    assert list(ov_tensor2.shape) == [1, 3, 24, 24]
    assert ov_tensor2.element_type == ov_tensor2.element_type
    assert np.shares_memory(ov_tensor1.data, ov_tensor2.data)
    assert np.array_equal(ov_tensor1.data[0:1, :, 24:, 24:], ov_tensor2.data)


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.bf16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_write_to_buffer(ov_type, numpy_dtype):
    ov_tensor = ov.Tensor(ov_type, ov.Shape([1, 3, 32, 32]))
    ones_arr = np.ones([1, 3, 32, 32], numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)


@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.bf16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
        (ov.Type.boolean, bool),
    ],
)
def test_set_shape(ov_type, numpy_dtype):
    shape = ov.Shape([1, 3, 32, 32])
    ref_shape = ov.Shape([1, 3, 48, 48])
    ref_shape_np = [1, 3, 28, 28]
    ov_tensor = ov.Tensor(ov_type, shape)

    ov_tensor.set_shape(ref_shape)
    assert list(ov_tensor.shape) == list(ref_shape)
    ov_tensor.shape = ref_shape
    assert list(ov_tensor.shape) == list(ref_shape)

    ones_arr = np.ones(list(ov_tensor.shape), numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)

    ov_tensor.set_shape(ref_shape_np)
    assert list(ov_tensor.shape) == ref_shape_np
    ov_tensor.shape = ref_shape_np
    assert list(ov_tensor.shape) == ref_shape_np

    zeros = np.zeros(ref_shape_np, numpy_dtype)
    ov_tensor.data[:] = zeros
    assert np.array_equal(ov_tensor.data, zeros)


@pytest.mark.parametrize(
    "ref_shape",
    [
        [1, 3, 24, 24],
        [1, 3, 32, 32],
    ],
)
def test_can_set_smaller_or_same_shape_on_preallocated_memory(ref_shape):
    ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    ones_arr = np.ascontiguousarray(ones_arr)
    ov_tensor = ov.Tensor(ones_arr, shared_memory=True)
    assert np.shares_memory(ones_arr, ov_tensor.data)
    ov_tensor.shape = ref_shape
    assert list(ov_tensor.shape) == ref_shape


def test_cannot_set_bigger_shape_on_preallocated_memory():
    ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    ones_arr = np.ascontiguousarray(ones_arr)
    ov_tensor = ov.Tensor(ones_arr, shared_memory=True)
    ref_shape = [1, 3, 48, 48]
    assert np.shares_memory(ones_arr, ov_tensor.data)
    with pytest.raises(RuntimeError) as e:
        ov_tensor.shape = ref_shape
    assert "failed" in str(e.value)


@pytest.mark.skip(reason="no support yet")
def test_can_reset_shape_after_decreasing_on_preallocated_memory():
    ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    ones_arr = np.ascontiguousarray(ones_arr)
    ov_tensor = ov.Tensor(ones_arr, shared_memory=True)
    ref_shape_1 = [1, 3, 24, 24]
    ref_shape_2 = [1, 3, 32, 32]
    assert np.shares_memory(ones_arr, ov_tensor.data)
    ov_tensor.shape = ref_shape_1
    assert list(ov_tensor.shape) == ref_shape_1
    ov_tensor.shape = ref_shape_2
    assert list(ov_tensor.shape) == ref_shape_2


def test_can_set_shape_other_dims():
    ov_tensor = ov.Tensor(np.float32, [1, 3, 48, 48])
    ref_shape_1 = [3, 28, 28]
    ov_tensor.shape = ref_shape_1
    assert list(ov_tensor.shape) == ref_shape_1


@pytest.mark.parametrize(
    "ov_type",
    [
        (ov.Type.u1),
        (ov.Type.u4),
        (ov.Type.i4),
    ],
)
def test_cannot_create_roi_from_packed_tensor(ov_type):
    ov_tensor = ov.Tensor(ov_type, [1, 3, 48, 48])
    with pytest.raises(RuntimeError) as e:
        ov.Tensor(ov_tensor, [0, 0, 24, 24], [1, 3, 48, 48])
    assert "for types with bitwidths less then 8 bit" in str(e.value)


@pytest.mark.parametrize(
    "ov_type",
    [
        (ov.Type.u1),
        (ov.Type.u4),
        (ov.Type.i4),
    ],
)
def test_cannot_get_strides_for_packed_tensor(ov_type):
    ov_tensor = ov.Tensor(ov_type, [1, 3, 48, 48])
    with pytest.raises(RuntimeError) as e:
        ov_tensor.get_strides()
    assert "Could not get strides for types with bitwidths less then 8 bit." in str(e.value)


@pytest.mark.parametrize(
    "dtype",
    [
        (np.uint8),
        (np.int8),
        (np.uint16),
        (np.uint32),
        (np.uint64),
    ],
)
@pytest.mark.parametrize(
    "ov_type",
    [
        (ov.Type.u1),
        (ov.Type.u4),
        (ov.Type.i4),
    ],
)
def test_init_with_packed_buffer(dtype, ov_type):
    shape = [1, 3, 32, 32]
    fit = np.dtype(dtype).itemsize * 8 / ov_type.bitwidth
    assert np.prod(shape) % fit == 0
    size = int(np.prod(shape) // fit)
    buffer = np.random.normal(size=size).astype(dtype)
    ov_tensor = ov.Tensor(buffer, shape, ov_type)
    assert ov_tensor.data.nbytes == ov_tensor.byte_size
    assert np.array_equal(ov_tensor.data.view(dtype), buffer)


@pytest.mark.parametrize(
    "shape",
    [
        ([1, 3, 28, 28]),
        ([1, 3, 27, 27]),
    ],
)
@pytest.mark.parametrize(
    ("low", "high", "ov_type", "dtype"),
    [
        (0, 2, ov.Type.u1, np.uint8),
        (0, 16, ov.Type.u4, np.uint8),
        (-8, 7, ov.Type.i4, np.int8),
        (0, 16, ov.Type.nf4, np.uint8),
    ],
)
def test_packing(shape, low, high, ov_type, dtype):
    ov_tensor = ov.Tensor(ov_type, shape)
    data = np.random.uniform(low, high, shape).astype(dtype)
    packed_data = pack_data(data, ov_tensor.element_type)
    ov_tensor.data[:] = packed_data
    unpacked = unpack_data(ov_tensor.data, ov_tensor.element_type, ov_tensor.shape)
    assert np.array_equal(unpacked, data)


@pytest.mark.parametrize(
    "dtype",
    [
        (np.uint8),
        (np.int8),
        (np.int16),
        (np.uint16),
        (np.int32),
        (np.uint32),
        (np.int64),
        (np.uint64),
        (np.float16),
        (np.float32),
        (np.float64),
    ],
)
@pytest.mark.parametrize(
    "element_type",
    [
        (ov.Type.u8),
        (ov.Type.i8),
        (ov.Type.i16),
        (ov.Type.u16),
        (ov.Type.i32),
        (ov.Type.u32),
        (ov.Type.i64),
        (ov.Type.u64),
    ],
)
def test_viewed_tensor(dtype, element_type):
    buffer = np.random.normal(size=(2, 16)).astype(dtype)
    fit = (dtype().nbytes * 8) / element_type.bitwidth
    tensor = ov.Tensor(buffer, (buffer.shape[0], int(buffer.shape[1] * fit)), element_type)
    assert np.array_equal(tensor.data, buffer.view(ov.utils.types.get_dtype(element_type)))


def test_viewed_tensor_default_type():
    buffer = np.random.normal(size=(2, 16))
    new_shape = (4, 8)
    tensor = ov.Tensor(buffer, new_shape)
    assert np.array_equal(tensor.data, buffer.reshape(new_shape))


def test_stride_calculation():
    data_type = np.float32
    arr = np.ones((16, 512, 1, 1)).astype(data_type)
    # Forces reorder of strides while keeping C-style memory.
    arr = arr.transpose((2, 0, 1, 3))
    ov_tensor = ov.Tensor(arr)
    assert ov_tensor is not None
    assert np.array_equal(ov_tensor.data, arr)

    elements = ov_tensor.shape[1] * ov_tensor.shape[2] * ov_tensor.shape[3]
    assert ov_tensor.strides[0] == elements * ov_tensor.get_element_type().size


@pytest.mark.parametrize(
    ("element_type", "dtype"),
    [
        (ov.Type.f32, np.float32),
        (ov.Type.f64, np.float64),
        (ov.Type.f16, np.float16),
        (ov.Type.bf16, np.float16),
        (ov.Type.i8, np.int8),
        (ov.Type.u8, np.uint8),
        (ov.Type.i32, np.int32),
        (ov.Type.u32, np.uint32),
        (ov.Type.i16, np.int16),
        (ov.Type.u16, np.uint16),
        (ov.Type.i64, np.int64),
        (ov.Type.u64, np.uint64),
    ],
)
def test_copy_to(dtype, element_type):
    tensor = ov.Tensor(shape=ov.Shape([3, 2, 2]), type=element_type)
    target_tensor = ov.Tensor(shape=ov.Shape([3, 2, 2]), type=element_type)

    ones_arr = np.ones(list(tensor.shape), dtype)
    tensor.data[:] = ones_arr

    zeros = np.zeros(list(target_tensor.shape), dtype)
    target_tensor.data[:] = zeros

    tensor.copy_to(target_tensor)
    assert tensor.shape == target_tensor.shape
    assert tensor.element_type == target_tensor.element_type
    assert tensor.byte_size == target_tensor.byte_size
    assert np.array_equal(tensor.data, target_tensor.data)


@pytest.mark.parametrize(
    "element_type",
    [
        (ov.Type.f32),
        (ov.Type.f64),
        (ov.Type.f16),
        (ov.Type.bf16),
        (ov.Type.i8),
        (ov.Type.u8),
        (ov.Type.i32),
        (ov.Type.u32),
        (ov.Type.i16),
        (ov.Type.u16),
        (ov.Type.i64),
        (ov.Type.u64),
    ],
)
def test_is_continuous(element_type):
    tensor = ov.Tensor(shape=ov.Shape([3, 2, 2]), type=element_type)
    assert tensor.is_continuous()


@pytest.mark.parametrize(
    "shared_flag",
    [
        (True),
        (False),
    ],
)
@pytest.mark.parametrize(
    "init_value",
    [
        (np.array([])),
        (np.array([], dtype=np.int32)),
        (np.empty(shape=(0))),
    ],
)
def test_init_from_empty_array(shared_flag, init_value):
    tensor = ov.Tensor(init_value, shared_memory=shared_flag)
    assert tensor.is_continuous()
    assert tuple(tensor.shape) == init_value.shape
    assert tensor.element_type.to_dtype() == init_value.dtype
    assert tensor.byte_size == init_value.nbytes
    assert np.array_equal(tensor.data, init_value)


@pytest.mark.parametrize(
    "init_value",
    [
        ([1.0, 2.0, 3.0]),
        ([21, 37, 42]),
        ([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]),
        ([[2.2, 6.5], [0.2, 6.7]]),
    ],
)
def test_init_from_list(init_value):
    tensor = ov.Tensor(init_value)
    assert np.array_equal(tensor.data, init_value)
    # Convert to numpy to perform all checks. Memory is not shared,
    # so it does not matter if data is stored in numpy format.
    _init_value = np.array(init_value)
    assert tuple(tensor.shape) == _init_value.shape
    assert tensor.element_type.to_dtype() == _init_value.dtype
    assert tensor.byte_size == _init_value.nbytes


def test_tensor_keeps_memory():
    def get_tensor():
        arr = np.ones((8, 16, 300), dtype=np.float32)
        assert sys.getrefcount(arr) == 2

        shared_tensor = ov.Tensor(arr, shared_memory=True)
        arr[0][0][0:2] = 0
        assert sys.getrefcount(arr) == 3

        del arr
        return shared_tensor

    tensor = get_tensor()
    assert np.allclose(tensor.data[0][0][0:3], [0, 0, 1])


@pytest.mark.parametrize(
    ("copy_func", "should_share_data"), [(copy, True), (deepcopy, False)]
)
def test_copy_and_deepcopy(copy_func, should_share_data):
    shape = (3, 4)
    value, outlier = 7, 100
    tensor_data = np.full(shape, value)
    tensor = ov.Tensor(tensor_data)
    tensor_copy = copy_func(tensor)

    assert np.array_equal(tensor_copy.data, tensor.data)
    assert tensor_copy is not tensor
    # Update value of the original tensor
    tensor.data[0, 0] = outlier
    assert tensor.data[0, 0] == outlier

    if should_share_data:
        assert tensor_copy.data[0, 0] == outlier
    else:
        assert tensor_copy.data[0, 0] == value


# supported dtypes by Pillow
@pytest.mark.parametrize(("numpy_dtype", "shape"), [
                         (np.float32, (224, 224)),
                         (np.int32, (224, 224)),
                         (np.uint8, (224, 224, 3)),
                         (np.uint16, (224, 224)),],)
def test_tensor_from_pillow(numpy_dtype, shape):
    from PIL import Image

    arr = generate_image(shape, numpy_dtype)
    img = Image.fromarray(arr)

    tensor = ov.Tensor(img)
    assert tensor.shape == shape
    assert tensor.element_type == ov.Type(numpy_dtype)
    assert isinstance(tensor.data, np.ndarray)
    assert tensor.data.dtype == numpy_dtype
    assert tensor.data.shape == shape
