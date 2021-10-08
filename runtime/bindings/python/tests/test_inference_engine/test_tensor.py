# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openvino import Tensor
import ngraph as ng


@pytest.mark.parametrize("ov_type, numpy_dtype", [
    (ng.impl.Type.f32, np.float32),
    (ng.impl.Type.f64, np.float64),
    (ng.impl.Type.f16, np.float16),
    (ng.impl.Type.bf16, np.float16),
    (ng.impl.Type.i8, np.int8),
    (ng.impl.Type.u8, np.uint8),
    (ng.impl.Type.i32, np.int32),
    (ng.impl.Type.u32, np.uint32),
    (ng.impl.Type.i16, np.int16),
    (ng.impl.Type.u16, np.uint16),
    (ng.impl.Type.i64, np.int64),
    (ng.impl.Type.u64, np.uint64),
    (ng.impl.Type.boolean, np.bool),
    (ng.impl.Type.u1, np.uint8),
])
def test_init_with_ngraph(ov_type, numpy_dtype):
    ov_tensors = []
    ov_tensors.append(Tensor(type=ov_type, shape=ng.impl.Shape([1, 3, 32, 32])))
    ov_tensors.append(Tensor(type=ov_type, shape = [1, 3, 32, 32]))
    for ov_tensor in ov_tensors:
        assert list(ov_tensor.shape) == [1, 3, 32, 32]
        assert ov_tensor.element_type == ov_type
        assert ov_tensor.data.dtype == numpy_dtype
        assert ov_tensor.data.shape == (1, 3, 32, 32)


@pytest.mark.parametrize("ov_type, numpy_dtype", [
    (ng.impl.Type.f32, np.float32),
    (ng.impl.Type.f64, np.float64),
    (ng.impl.Type.f16, np.float16),
    (ng.impl.Type.i8, np.int8),
    (ng.impl.Type.u8, np.uint8),
    (ng.impl.Type.i32, np.int32),
    (ng.impl.Type.u32, np.uint32),
    (ng.impl.Type.i16, np.int16),
    (ng.impl.Type.u16, np.uint16),
    (ng.impl.Type.i64, np.int64),
    (ng.impl.Type.u64, np.uint64),
    (ng.impl.Type.boolean, np.bool)
])
def test_init_with_numpy(ov_type, numpy_dtype):
    shape = (1, 3, 127, 127)
    ov_shape = ng.impl.Shape(shape)
    ones_arr = np.ones(shape, numpy_dtype)
    ones_ov_tensor = Tensor(array=ones_arr)
    ov_tensors = []
    ov_tensors.append(Tensor(dtype=numpy_dtype, shape=shape))
    ov_tensors.append(Tensor(dtype=np.dtype(numpy_dtype), shape=shape))
    ov_tensors.append(Tensor(dtype=np.dtype(numpy_dtype), shape=np.array(shape)))
    ov_tensors.append(ones_ov_tensor)
    ov_tensors.append(Tensor(ones_arr, ones_arr.strides))
    ov_tensors.append(Tensor(dtype=numpy_dtype, shape=ov_shape))
    ov_tensors.append(Tensor(dtype=np.dtype(numpy_dtype), shape=ov_shape))
    for ov_tensor in ov_tensors:
        assert tuple(ov_tensor.shape) == shape
        assert ov_tensor.element_type == ov_type
        assert isinstance(ov_tensor.data, np.ndarray)
        assert ov_tensor.data.dtype == numpy_dtype
        assert ov_tensor.data.shape == shape

    assert np.shares_memory(ones_arr, ones_ov_tensor.data)
    assert np.array_equal(ones_ov_tensor.data, ones_arr)


def test_init_with_roi_tensor():
    array = np.random.normal(size=[1, 3, 48, 48])
    ov_tensor1 = Tensor(array)
    ov_tensor2 = Tensor(ov_tensor1, [0, 0, 24, 24], [1, 1, 48, 48])
    assert list(ov_tensor2.shape) == [1, 1, 24, 24]
    assert ov_tensor2.element_type == ov_tensor2.element_type
    assert np.shares_memory(ov_tensor2.data, ov_tensor1.data)

    ov_tensor3 = Tensor(ov_tensor1, ng.impl.Coordinate([0, 0, 16, 16]), ng.impl.Coordinate([1, 2, 48, 48]))
    assert list(ov_tensor3.shape) == [1, 2, 32, 32]
    assert ov_tensor3.element_type == ov_tensor1.element_type
    assert np.shares_memory(ov_tensor3.data, ov_tensor1.data)


@pytest.mark.parametrize("ov_type, numpy_dtype", [
    (ng.impl.Type.f32, np.float32),
    (ng.impl.Type.f64, np.float64),
    (ng.impl.Type.f16, np.float16),
    (ng.impl.Type.bf16, np.float16),
    (ng.impl.Type.i8, np.int8),
    (ng.impl.Type.u8, np.uint8),
    (ng.impl.Type.i32, np.int32),
    (ng.impl.Type.u32, np.uint32),
    (ng.impl.Type.i16, np.int16),
    (ng.impl.Type.u16, np.uint16),
    (ng.impl.Type.i64, np.int64),
    (ng.impl.Type.u64, np.uint64),
    (ng.impl.Type.boolean, np.bool),
    (ng.impl.Type.u1, np.uint8),
])
def test_write_to_buffer(ov_type, numpy_dtype):
    ov_tensor = Tensor(ov_type, ng.impl.Shape([1, 3, 32, 32]))
    ones_arr = np.ones([1, 3, 32, 32], numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)


@pytest.mark.parametrize("ov_type, numpy_dtype", [
    (ng.impl.Type.f32, np.float32),
    (ng.impl.Type.f64, np.float64),
    (ng.impl.Type.f16, np.float16),
    (ng.impl.Type.bf16, np.float16),
    (ng.impl.Type.i8, np.int8),
    (ng.impl.Type.u8, np.uint8),
    (ng.impl.Type.i32, np.int32),
    (ng.impl.Type.u32, np.uint32),
    (ng.impl.Type.i16, np.int16),
    (ng.impl.Type.u16, np.uint16),
    (ng.impl.Type.i64, np.int64),
    (ng.impl.Type.u64, np.uint64),
    (ng.impl.Type.boolean, np.bool),
    (ng.impl.Type.u1, np.uint8),
])
def test_set_shape(ov_type, numpy_dtype):
    shape = ng.impl.Shape([1, 3, 32, 32])
    ref_shape = ng.impl.Shape([1, 3, 48, 48])
    ref_shape_np = [1, 3, 28, 28]
    ov_tensor = Tensor(ov_type, shape)
    ov_tensor.shape = ref_shape
    assert list(ov_tensor.shape) == list(ref_shape)
    ones_arr = np.ones(list(ov_tensor.shape), numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)
    ov_tensor.shape = ref_shape_np
    assert list(ov_tensor.shape) == ref_shape_np
    zeros = np.zeros(ref_shape_np, numpy_dtype)
    ov_tensor.data[:] = zeros
    assert np.array_equal(ov_tensor.data, zeros)


def test_cannot_set_shape_on_preallocated_memory():
    ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    ov_tensor = Tensor(ones_arr)
    with pytest.raises(RuntimeError) as e:
        ov_tensor.shape = ng.impl.Shape([1, 3, 48, 48])
    assert "Cannot call setShape for Blobs created on top of preallocated memory" in str(e.value)


def test_cannot_set_shape_incorrect_dims():
    ov_tensor = Tensor(np.float32, [1, 3, 48, 48])
    with pytest.raises(RuntimeError) as e:
        ov_tensor.shape = [3, 28, 28]
    assert "Dims and format are inconsistent" in str(e.value)
