# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime as ov
import openvino.runtime.opset13 as ops
from openvino.runtime import Type
from openvino.runtime.op import Constant
from openvino.helpers import pack_data, unpack_data

import pytest
from enum import Enum


class DataGetter(Enum):
    COPY = 1
    VIEW = 2


@pytest.mark.parametrize(
    ("src_dtype"),
    [
        (np.float16),
        (np.float32),
        (np.float64),
        (np.int8),
        (np.uint8),
        (np.int16),
        (np.uint16),
        (np.int32),
        (np.uint32),
        (np.int64),
        (bool),
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("dst_dtype"),
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
        (ov.Type.boolean),
        (ov.Type.u1),
        (ov.Type.u4),
        (ov.Type.i4),
        (np.float16),
        (np.float32),
        (np.float64),
        (np.int8),
        (np.uint8),
        (np.int16),
        (np.uint16),
        (np.int32),
        (np.uint32),
        (np.int64),
        (bool),
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
@pytest.mark.parametrize(
    ("data_getter"),
    [
        (DataGetter.COPY),
        (DataGetter.VIEW),
    ],
)
def test_init_with_array(src_dtype, dst_dtype, shared_flag, data_getter):
    data = np.random.rand(1, 2, 16, 8) + 0.5
    data = data.astype(src_dtype)
    # C-style data for shared memory
    if shared_flag is True:
        data = np.ascontiguousarray(data)
    # Create constant from based on numpy dtype or openvino type
    ov_const = ops.constant(data, dtype=dst_dtype, shared_memory=shared_flag)
    # Convert to dtype if OpenVINO Type
    _dst_dtype = dst_dtype.to_dtype() if isinstance(dst_dtype, Type) else dst_dtype
    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(list(ov_const.shape) == [1, 2, 16, 8])
    assert ov_const.get_element_type().to_dtype() == _dst_dtype
    # Cast original data and compare values to Constant
    expected_result = data.astype(_dst_dtype)
    if data_getter == DataGetter.COPY:
        const_data = ov_const.get_data()
    elif data_getter == DataGetter.VIEW:
        const_data = ov_const.data
    else:
        raise AttributeError("Unknown DataGetter passed!")
    assert const_data.dtype == _dst_dtype
    assert np.allclose(const_data, expected_result)


@pytest.mark.parametrize(
    ("init_value"),
    [
        (1.5),  # float
        (2),  # int
    ],
)
@pytest.mark.parametrize(
    ("src_dtype"),
    [
        (float),
        (int),
        (np.float16),
        (np.float32),
        (np.float64),
        (np.int8),
        (np.uint8),
        (np.int16),
        (np.uint16),
        (np.int32),
        (np.uint32),
        (np.int64),
        (bool),
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("dst_dtype"),
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
        (ov.Type.boolean),
        (ov.Type.u1),
        (ov.Type.u4),
        (ov.Type.i4),
        (np.float16),
        (np.float32),
        (np.float64),
        (np.int8),
        (np.uint8),
        (np.int16),
        (np.uint16),
        (np.int32),
        (np.uint32),
        (np.int64),
        (bool),
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
@pytest.mark.parametrize(
    ("data_getter"),
    [
        (DataGetter.COPY),
        (DataGetter.VIEW),
    ],
)
def test_init_with_scalar(init_value, src_dtype, dst_dtype, shared_flag, data_getter):
    data = src_dtype(init_value)
    # Create constant from based on numpy dtype or openvino type
    ov_const = ops.constant(data, dtype=dst_dtype, shared_memory=shared_flag)
    # Convert to dtype if OpenVINO Type
    _dst_dtype = dst_dtype.to_dtype() if isinstance(dst_dtype, Type) else dst_dtype
    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(list(ov_const.shape) == [])
    assert ov_const.get_element_type().to_dtype() == _dst_dtype
    # Cast original data and compare values to Constant
    expected_result = np.array(data).astype(_dst_dtype)
    if data_getter == DataGetter.COPY:
        const_data = ov_const.get_data()
    elif data_getter == DataGetter.VIEW:
        const_data = ov_const.data
    else:
        raise AttributeError("Unknown DataGetter passed!")
    assert const_data.dtype == _dst_dtype
    assert np.allclose(const_data, expected_result)


@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
def test_write_to_buffer(shared_flag):
    arr_0 = np.ones([1, 3, 32, 32])
    ov_const = ops.constant(arr_0, shared_memory=shared_flag)
    arr_1 = np.ones([1, 3, 32, 32]) + 1
    ov_const.data[:] = arr_1
    assert np.array_equal(ov_const.data, arr_1)


@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
def test_memory_sharing(shared_flag):
    arr = np.ones([1, 3, 32, 32])
    ov_const = ops.constant(arr, shared_memory=shared_flag)
    arr += 1
    if shared_flag is True:
        assert np.array_equal(ov_const.data, arr)
    else:
        assert not np.array_equal(ov_const.data, arr)
