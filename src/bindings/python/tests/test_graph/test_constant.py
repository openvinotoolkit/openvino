# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov

import openvino.runtime.opset13 as ops
from openvino import Type, PartialShape, Model, Tensor, compile_model
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
        (Type.f32),
        (Type.f64),
        (Type.f16),
        (Type.i8),
        (Type.u8),
        (Type.i32),
        (Type.u32),
        (Type.i16),
        (Type.u16),
        (Type.i64),
        (Type.u64),
        (Type.boolean),
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
    ov_const = ops.constant(data, dst_dtype, shared_memory=shared_flag)

    # Check shape and element type of Constant class
    assert isinstance(ov_const, Constant)
    assert np.all(tuple(ov_const.shape) == data.shape)
    # Additionally check if Constant type matches dst_type if Type was passed:
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
        (Type.f32),
        (Type.f64),
        (Type.f16),
        (Type.i8),
        (Type.u8),
        (Type.i32),
        (Type.u32),
        (Type.i16),
        (Type.u16),
        (Type.i64),
        (Type.u64),
        (Type.boolean),
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
    # Additionally check if Constant type matches dst_type if Type was passed:
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
def test_init_bf16_populate(src_dtype, shared_flag, data_getter):
    data = np.random.rand(1, 2, 16, 8) + 0.5
    data = data.astype(src_dtype)

    # To create bf16 constant, allocate memory and populate it:
    init_data = np.zeros(shape=data.shape, dtype=src_dtype)
    ov_const = ops.constant(init_data, dtype=Type.bf16, shared_memory=shared_flag)
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
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.i32, np.int32),
        (Type.i16, np.int16),
    ],
)
@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
def test_init_bf16_direct(ov_type, numpy_dtype, shared_flag):
    data = np.random.rand(4) + 1.5
    data = data.astype(numpy_dtype)

    bf16_const = ops.constant(data, dtype=Type.bf16, shared_memory=shared_flag, name="bf16_constant")
    convert = ops.convert(bf16_const, data.dtype)
    parameter = ops.parameter(PartialShape([-1]), ov_type)
    add_op = ops.add(parameter, convert)
    model = Model([add_op], [parameter])

    compiled = compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    assert np.allclose(data, result, rtol=0.01)


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
        (0, 2, Type.u1, np.uint8),
        (0, 16, Type.u4, np.uint8),
        (-8, 7, Type.i4, np.int8),
        (0, 16, Type.nf4, np.uint8),
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
def test_constant_helper_packing(shape, low, high, ov_type, src_dtype, shared_flag, data_getter):
    data = np.random.uniform(low, high, shape).astype(src_dtype)

    # Allocate memory first:
    ov_const = ops.constant(np.zeros(shape=data.shape, dtype=src_dtype),
                            dtype=ov_type,
                            shared_memory=shared_flag)
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
    ("ov_type", "src_dtype"),
    [
        (Type.u1, np.uint8),
        (Type.u4, np.uint8),
        (Type.i4, np.int8),
        (Type.nf4, np.uint8),
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
def test_constant_direct_packing(ov_type, src_dtype, shared_flag, data_getter):
    data = np.ones((2, 4, 16)).astype(src_dtype)

    ov_const = ops.constant(data, dtype=ov_type, shared_memory=shared_flag)

    # Always unpack the data!
    if data_getter == DataGetter.COPY:
        unpacked = unpack_data(ov_const.get_data(), ov_const.get_element_type(), ov_const.shape)
        assert np.array_equal(unpacked, data)
    elif data_getter == DataGetter.VIEW:
        unpacked = unpack_data(ov_const.data, ov_const.get_element_type(), ov_const.shape)
        assert np.array_equal(unpacked, data)
    else:
        raise AttributeError("Unknown DataGetter passed!")

    assert np.array_equal(unpacked, data)
    assert not np.shares_memory(unpacked, data)


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


OPSETS = [ov.runtime.opset12, ov.runtime.opset13]


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e5m2_constant(ov_type, numpy_dtype, opset):
    data = np.array([4.75, 4.5, -5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, -0.1, -0.2, -0.3,
                     -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0.0000152587890625, 448, 500, 512, 57344], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov.Type.f8e5m2, name="f8e5m2_constant")
    convert = opset.convert(compressed_const, data.dtype)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [5.0, 4.0, -5.0, 0.0, 0.09375, 0.1875, 0.3125, 0.375, 0.5, 0.625, 0.75,
              0.75, 0.875, 1.0, -0.0, -0.09375, -0.1875, -0.3125, -0.375,
              -0.5, -0.625, -0.75, -0.75, -0.875, -1.0, 0.0000152587890625,
              448, 512, 512, 57344]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e4m3_constant(ov_type, numpy_dtype, opset):
    data = np.array([4.75, 4.5, -5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, -0.1, -0.2, -0.3,
                     -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1, 448, 512], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov.Type.f8e4m3, name="f8e4m3_constant")
    convert = opset.convert(compressed_const, data.dtype)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [5.0, 4.5, -5.0, 0.0, 0.1015625, 0.203125, 0.3125,
              0.40625, 0.5, 0.625, 0.6875, 0.8125, 0.875, 1,
              -0, -0.1015625, -0.203125, -0.3125, -0.40625, -0.5, -0.625,
              -0.6875, -0.8125, -0.875, -1, 448, np.nan]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target, equal_nan=True)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e8m0_constant_matrix(ov_type, numpy_dtype, opset):
    pytest.skip("CVS-145281 BUG: nan to inf repro.")

    shape = (2, 2)
    data = np.full(shape, np.nan)

    compressed_const = opset.constant(data, dtype=ov_type, name="fx_constant")
    convert_to_fp8 = opset.convert(compressed_const, Type.f8e8m0)
    convert_back = opset.convert(convert_to_fp8, ov_type)
    parameter = opset.parameter(ov.PartialShape([-1, -1]), ov_type)
    add_op = opset.add(parameter, convert_back)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model, "GPU")
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = np.full(shape, np.nan)

    assert np.allclose(result, target, equal_nan=True)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e8m0_constant_single_nan(ov_type, numpy_dtype, opset):
    pytest.skip("CVS-145281 BUG: nan to inf repro.")

    data = np.array([np.nan], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov.Type.f8e8m0, name="f8e8m0_constant")
    convert = opset.convert(compressed_const, data.dtype)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [np.nan]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target, equal_nan=True)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e8m0_constant(ov_type, numpy_dtype, opset):
    pytest.skip("CVS-145281 BUG: nan to inf repro. [random - depends on the device]")

    data = np.array([4.75, 4.5, 5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, 1.1, 1.2, 1.3,
                     1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 448, 512, np.nan], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov.Type.f8e8m0, name="f8e8m0_constant")
    convert = opset.convert(compressed_const, data.dtype)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [4.0, 4.0, 4.0, 0.0, 0.125, 0.25, 0.25,
              0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
              0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
              2.0, 2.0, 2.0, 2.0, 512, 512, np.nan]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target, equal_nan=True)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e5m2_convert(ov_type, numpy_dtype, opset):
    data = np.array([4.75, 4.5, -5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, -0.1, -0.2, -0.3,
                     -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0.0000152587890625, 448, 500, 512, 57344], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov_type, name="fx_constant")
    convert_to_fp8 = opset.convert(compressed_const, Type.f8e5m2)
    convert_back = opset.convert(convert_to_fp8, ov_type)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert_back)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [5.0, 4.0, -5.0, 0.0, 0.09375, 0.1875, 0.3125, 0.375, 0.5, 0.625, 0.75,
              0.75, 0.875, 1.0, -0.0, -0.09375, -0.1875, -0.3125, -0.375,
              -0.5, -0.625, -0.75, -0.75, -0.875, -1.0, 0.0000152587890625,
              448, 512, 512, 57344]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e4m3_convert(ov_type, numpy_dtype, opset):
    data = np.array([4.75, 4.5, -5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, -0.1, -0.2, -0.3,
                     -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1, 448, 512], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov_type, name="fx_constant")
    convert_to_fp8 = opset.convert(compressed_const, Type.f8e4m3)
    convert_back = opset.convert(convert_to_fp8, ov_type)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert_back)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [5.0, 4.5, -5.0, 0.0, 0.1015625, 0.203125, 0.3125,
              0.40625, 0.5, 0.625, 0.6875, 0.8125, 0.875, 1,
              -0, -0.1015625, -0.203125, -0.3125, -0.40625, -0.5, -0.625,
              -0.6875, -0.8125, -0.875, -1, 448, np.nan]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target, equal_nan=True)


@pytest.mark.parametrize(("opset"), OPSETS)
@pytest.mark.parametrize(
    ("ov_type", "numpy_dtype"),
    [
        (Type.f32, np.float32),
        (Type.f16, np.float16),
    ],
)
def test_float_to_f8e8m0_convert(ov_type, numpy_dtype, opset):
    pytest.skip("CVS-145281 BUG: nan to inf repro. [random - depends on the device]")

    data = np.array([4.75, 4.5, 5.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1, -0.0, 1.1, 1.2, 1.3,
                     1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 448, 512, np.nan], dtype=numpy_dtype)

    compressed_const = opset.constant(data, dtype=ov_type, name="fx_constant")
    convert_to_fp8 = opset.convert(compressed_const, Type.f8e8m0)
    convert_back = opset.convert(convert_to_fp8, ov_type)
    parameter = opset.parameter(ov.PartialShape([-1]), ov_type)
    add_op = opset.add(parameter, convert_back)
    model = ov.Model([add_op], [parameter])

    compiled = ov.compile_model(model)
    tensor = np.zeros(data.shape, dtype=numpy_dtype)
    result = compiled(tensor)[0]

    target = [4.0, 4.0, 4.0, 0.0, 0.125, 0.25, 0.25,
              0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
              0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
              2.0, 2.0, 2.0, 2.0, 512, 512, np.nan]
    target = np.array(target, dtype=numpy_dtype)

    assert np.allclose(result, target, equal_nan=True)


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
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("dst_dtype"),
    [
        (None),
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
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("copy_flag"),
    [
        (True),
        (False),
    ],
)
def test_get_data_casting(src_dtype, dst_dtype, copy_flag):
    data = np.random.rand(2, 4, 16) + 0.01  # do not allow 0s -- extra edge-case for bool type
    data = data.astype(src_dtype)

    ov_const = ops.constant(data, dtype=src_dtype)
    arr = ov_const.get_data(dtype=dst_dtype, copy=copy_flag)

    if (src_dtype == dst_dtype or dst_dtype is None) and copy_flag is False:
        assert arr.flags["OWNDATA"] is False
        assert np.array_equal(arr, data)
    else:
        assert arr.flags["OWNDATA"] is True
        assert np.array_equal(arr, data.astype(dst_dtype))


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
        (np.bool_),
    ],
)
@pytest.mark.parametrize(
    ("copy_flag"),
    [
        (True),
        (False),
    ],
)
def test_get_data_casting_bool(src_dtype, copy_flag):
    data = np.array([1.0, 0.0, 2.0, 0.5, 0.3, 0.1, 3.0]).astype(src_dtype)

    ov_const = ops.constant(data, dtype=src_dtype)
    arr = ov_const.get_data(dtype=np.bool_, copy=copy_flag)

    if src_dtype == np.bool_ and copy_flag is False:
        assert arr.flags["OWNDATA"] is False
        assert np.array_equal(arr, data)
    else:
        assert arr.flags["OWNDATA"] is True
        assert np.array_equal(arr, data.astype(np.bool_))


@pytest.mark.parametrize(
    ("src_dtype"),
    [
        (np.float16),
        (np.float32),
        (np.float64),
    ],
)
@pytest.mark.parametrize(
    ("dst_dtype"),
    [
        (None),
        (np.float16),
        (np.float32),
        (np.float64),
    ],
)
@pytest.mark.parametrize(
    ("copy_flag"),
    [
        (True),
        (False),
    ],
)
def test_get_data_casting_bf16(src_dtype, dst_dtype, copy_flag):
    data = np.array([1.0, 0.0, 1012.5, 0.5, 2.0]).astype(src_dtype)
    ov_const = ops.constant(data, dtype=Type.bf16)

    arr = ov_const.get_data(dtype=dst_dtype, copy=copy_flag)

    expected_result = np.array([1.0, 0.0, 1012.0, 0.5, 2.0], dtype=np.float32)

    if dst_dtype is None and copy_flag is False:
        assert arr.flags["OWNDATA"] is False
        assert arr.dtype == np.float16
        assert np.array_equal(arr.view(np.int16), expected_result.view(np.int16)[1::2])
    elif dst_dtype == np.float16 and copy_flag is False:
        assert arr.flags["OWNDATA"] is False
        assert np.array_equal(arr.view(np.int16), expected_result.view(np.int16)[1::2])
    else:  # copy_flag is True
        assert arr.flags["OWNDATA"] is True
        if dst_dtype in [None, np.float16]:
            assert np.array_equal(arr.view(np.int16), expected_result.view(np.int16)[1::2])
        else:  # up-casting to np.float32 or np.float64
            assert np.array_equal(arr, expected_result)


@pytest.mark.parametrize(
    ("src_dtype"),
    [
        (np.int8),
    ],
)
@pytest.mark.parametrize(
    ("ov_type"),
    [
        (Type.u1),
    ],
)
@pytest.mark.parametrize(
    ("dst_dtype"),
    [
        (None),
        (np.int8),
    ],
)
@pytest.mark.parametrize(
    ("copy_flag"),
    [
        (True),
        (False),
    ],
)
def test_get_data_casting_packed(src_dtype, ov_type, dst_dtype, copy_flag):
    data = np.array([[0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=src_dtype)
    ov_const = ops.constant(value=data, dtype=ov_type)
    arr = ov_const.get_data(dtype=dst_dtype, copy=copy_flag)

    if dst_dtype is None:
        if copy_flag:
            assert arr.flags["OWNDATA"] is True
        else:
            assert arr.flags["OWNDATA"] is False
        assert np.array_equal(arr, np.packbits(data))
    else:
        assert arr.flags["OWNDATA"] is True
        assert np.array_equal(arr, data)


@pytest.mark.parametrize(
    ("shared_flag"),
    [
        (True),
        (False),
    ],
)
def test_const_from_tensor(shared_flag):
    shape = [1, 3, 32, 32]
    arr = np.ones(shape).astype(np.float32)
    ov_tensor = Tensor(arr, shape, Type.f32)
    ov_const = ops.constant(tensor=ov_tensor, shared_memory=shared_flag)

    assert isinstance(ov_const, Constant)
    assert np.all(list(ov_const.shape) == shape)
    arr += 1

    if shared_flag is True:
        assert np.array_equal(ov_const.data, arr)
        assert np.shares_memory(arr, ov_const.data)
    else:
        assert not np.array_equal(ov_const.data, arr)
        assert not np.shares_memory(arr, ov_const.data)
