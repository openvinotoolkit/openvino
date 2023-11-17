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
        (ov.Type.i8),
        (ov.Type.u8),
        (ov.Type.i32),
        (ov.Type.u32),
        (ov.Type.i16),
        (ov.Type.u16),
        (ov.Type.i64),
        (ov.Type.u64),
        (ov.Type.boolean),
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

    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(tuple(ov_const.shape) == data.shape)
    # Additionally check if Constant type matches dst_type if ov.Type was passed:
    if isinstance(dst_dtype, Type):
            assert ov_const.get_element_type() == dst_dtype
    # Convert to dtype if OpenVINO Type
    _dst_dtype = dst_dtype.to_dtype() if isinstance(dst_dtype, Type) else dst_dtype
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
        (ov.Type.i8),
        (ov.Type.u8),
        (ov.Type.i32),
        (ov.Type.u32),
        (ov.Type.i16),
        (ov.Type.u16),
        (ov.Type.i64),
        (ov.Type.u64),
        (ov.Type.boolean),
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

    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(list(ov_const.shape) == [])
    # Additionally check if Constant type matches dst_type if ov.Type was passed:
    if isinstance(dst_dtype, Type):
            assert ov_const.get_element_type() == dst_dtype
    # Convert to dtype if OpenVINO Type
    _dst_dtype = dst_dtype.to_dtype() if isinstance(dst_dtype, Type) else dst_dtype
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
    ("src_dtype"),
    [
        (np.float16),
        (np.uint16),
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
def test_init_bf16(src_dtype, shared_flag, data_getter):
    data = np.random.rand(1, 2, 16, 8) + 0.5
    data = data.astype(src_dtype)

    # To create bf16 constant, allocate memory and populate it:
    ov_const = ops.constant(np.zeros(shape=data.shape, dtype=src_dtype), dtype=Type.bf16, shared_memory=shared_flag)
    ov_const.data[:] = data

    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(list(ov_const.shape) == [1, 2, 16, 8])
    assert ov_const.get_element_type() == Type.bf16

    _dst_dtype = Type.bf16.to_dtype()

    assert ov_const.get_element_type().to_dtype() == _dst_dtype
    # Compare values to Constant
    if data_getter == DataGetter.COPY:
        const_data = ov_const.get_data()
    elif data_getter == DataGetter.VIEW:
        const_data = ov_const.data
    else:
        raise AttributeError("Unknown DataGetter passed!")
    assert const_data.dtype == _dst_dtype
    assert np.allclose(const_data, data)


@pytest.mark.parametrize(
    "shape",
    [
        ([1, 3, 28, 28]),
        ([1, 3, 27, 27]),
    ],
)
@pytest.mark.parametrize(
    ("low", "high", "ov_type", "src_dtype"),
    [
        (0, 2, ov.Type.u1, np.uint8),
        (0, 16, ov.Type.u4, np.uint8),
        (-8, 7, ov.Type.i4, np.int8),
        (0, 16, ov.Type.nf4, np.uint8),
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
def test_constant_packing(shape, low, high, ov_type, src_dtype, shared_flag, data_getter):
    data = np.random.uniform(low, high, shape).astype(src_dtype)

    # Allocate memory first:
    ov_const = ops.constant(
                    np.zeros(shape=data.shape, dtype=src_dtype),
                    dtype=ov_type,
                    shared_memory=shared_flag,
               )
    # Fill data with packed values
    packed_data = pack_data(data, ov_const.get_element_type())
    ov_const.data[:] = packed_data

    # Always unpack the data!
    if data_getter == DataGetter.COPY:
        unpacked = unpack_data(ov_const.get_data(), ov_const.get_element_type(), ov_const.shape)
    elif data_getter == DataGetter.VIEW:
        unpacked = unpack_data(ov_const.data, ov_const.get_element_type(), ov_const.shape)
    else:
        raise AttributeError("Unknown DataGetter passed!")

    assert np.array_equal(unpacked, data)


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
        assert np.shares_memory(arr, ov_const.data)
    else:
        assert not np.array_equal(ov_const.data, arr)
        assert not np.shares_memory(arr, ov_const.data)


@pytest.mark.parametrize(
    ("ov_type", "src_dtype"),
    [
        (Type.u1, np.uint8),
        (Type.u4, np.uint8),
        (Type.i4, np.int8),
        (Type.nf4, np.uint8),
        (Type.bf16, np.float16),
    ],
)
def test_raise_for_packed_types(ov_type, src_dtype):
    data = np.ones((2, 4, 16)).astype(src_dtype)

    with pytest.raises(RuntimeError) as err:
        _ = ops.constant(data, dtype=ov_type)

    assert "All values must be equal to 0 to initialize Constant with type of" in str(err.value) 
