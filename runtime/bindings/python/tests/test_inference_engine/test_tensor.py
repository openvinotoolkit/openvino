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
    ov_tensor = Tensor(ov_type, ng.impl.Shape([1, 3, 32, 32]))
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
    ones_arr = np.ones(shape=(1, 3, 127, 127), dtype=numpy_dtype)
    ov_tensor = Tensor(ones_arr)
    assert ov_tensor.element_type == ov_type
    assert isinstance(ov_tensor.data, np.ndarray)
    assert ov_tensor.data.dtype == numpy_dtype
    assert ov_tensor.data.shape == (1, 3, 127, 127)
    assert np.shares_memory(ones_arr, ov_tensor.data)
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
    ref_shape_np = (1, 3, 48, 48)
    ov_tensor = Tensor(ov_type, shape)
    ov_tensor.shape = ref_shape
    assert ov_tensor.data.shape == ref_shape_np
    ones_arr = np.ones(ref_shape_np, numpy_dtype)
    ov_tensor.data[:] = ones_arr
    assert np.array_equal(ov_tensor.data, ones_arr)


def test_cannot_set_shape_on_preallocated_memory():
    ones_arr = np.ones(shape=(1, 3, 32, 32), dtype=np.float32)
    ov_tensor = Tensor(ones_arr)
    with pytest.raises(RuntimeError) as e:
        ov_tensor.shape = ng.impl.Shape([1, 3, 48, 48])
    assert "Blob::setShape requires dense blob" in str(e.value)
