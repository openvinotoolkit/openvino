# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import Tensor, Type
from openvino.op import Constant

from tests.utils.helpers import generate_image


@pytest.mark.parametrize(("cls", "cls_str"), [
    (Tensor, "TENSOR"),
    (Constant, "CONSTANT"),
])
def test_init_with_numpy_fail(cls, cls_str):
    arr = np.asfortranarray(generate_image())  # F-style array

    with pytest.raises(RuntimeError) as e:
        _ = cls(array=arr, shared_memory=True)

    assert "SHARED MEMORY MODE FOR THIS " + cls_str + " IS NOT APPLICABLE!" in str(e.value)


@pytest.mark.parametrize("cls", [Tensor, Constant])
@pytest.mark.parametrize("shared_flag", [True, False])
@pytest.mark.parametrize(("ov_type", "numpy_dtype"), [
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.f16, np.float16),
    (Type.i8, np.int8),
    (Type.u8, np.uint8),
    (Type.i32, np.int32),
    (Type.u32, np.uint32),
    (Type.i16, np.int16),
    (Type.u16, np.uint16),
    (Type.i64, np.int64),
    (Type.u64, np.uint64),
    (Type.boolean, bool),
])
def test_with_numpy_memory(cls, shared_flag, ov_type, numpy_dtype):
    arr = np.ascontiguousarray(generate_image().astype(numpy_dtype))
    ov_object = cls(array=arr, shared_memory=shared_flag)

    assert ov_object.get_element_type() == ov_type
    assert tuple(ov_object.shape) == arr.shape

    assert isinstance(ov_object.data, np.ndarray)
    assert ov_object.data.dtype == numpy_dtype
    assert ov_object.data.shape == arr.shape
    assert np.array_equal(ov_object.data, arr)

    if shared_flag is True:
        assert np.shares_memory(arr, ov_object.data)
    else:
        assert not (np.shares_memory(arr, ov_object.data))


@pytest.mark.parametrize("cls", [Tensor, Constant])
@pytest.mark.parametrize("shared_flag", [True, False])
def test_with_external_memory(cls, shared_flag):
    class ArrayLikeObject:
        # Array-like object to test inputs similar to torch.Tensor and tf.Tensor
        def __init__(self, array) -> None:
            self.data = array

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

        def to_numpy(self):
            return self.data

    external_object = ArrayLikeObject(np.ascontiguousarray(generate_image()))
    ov_object = cls(array=external_object.to_numpy(), shared_memory=shared_flag)

    assert np.array_equal(ov_object.data.dtype, external_object.dtype)
    assert np.array_equal(ov_object.data.shape, external_object.shape)
    assert np.array_equal(ov_object.data, external_object.to_numpy())

    if shared_flag is True:
        assert np.shares_memory(external_object.to_numpy(), ov_object.data)
    else:
        assert not (np.shares_memory(external_object.to_numpy(), ov_object.data))


@pytest.mark.parametrize("cls", [Constant])
@pytest.mark.parametrize("shared_flag_one", [True, False])
@pytest.mark.parametrize("shared_flag_two", [True, False])
@pytest.mark.parametrize(("ov_type", "numpy_dtype"), [
    (Type.f32, np.float32),
    (Type.f64, np.float64),
    (Type.f16, np.float16),
    (Type.i8, np.int8),
    (Type.u8, np.uint8),
    (Type.i32, np.int32),
    (Type.u32, np.uint32),
    (Type.i16, np.int16),
    (Type.u16, np.uint16),
    (Type.i64, np.int64),
    (Type.u64, np.uint64),
    (Type.boolean, bool),
])
def test_with_tensor_memory(cls, shared_flag_one, shared_flag_two, ov_type, numpy_dtype):
    arr = np.ascontiguousarray(generate_image().astype(numpy_dtype))
    ov_tensor = Tensor(arr, shared_memory=shared_flag_one)
    ov_object = cls(tensor=ov_tensor, shared_memory=shared_flag_two)

    # Case 1: all data is shared
    if shared_flag_one is True and shared_flag_two is True:
        assert np.shares_memory(arr, ov_object.data)
        assert np.shares_memory(ov_tensor.data, ov_object.data)
    # Case 2: data is shared only between object and Tensor
    elif shared_flag_one is False and shared_flag_two is True:
        assert not (np.shares_memory(arr, ov_object.data))
        assert np.shares_memory(ov_tensor.data, ov_object.data)
    # Case 3: data is not shared, copy occurs in the object's constructor
    else:
        assert not (np.shares_memory(arr, ov_object.data))
        assert not (np.shares_memory(ov_tensor.data, ov_object.data))


@pytest.mark.parametrize("cls", [Tensor, Constant])
@pytest.mark.parametrize("shared_flag", [True, False])
@pytest.mark.parametrize("scalar", [
    np.array(2),
    np.array(1.0),
    np.float32(3.0),
    np.int64(7.0),
    4,
    5.0,
])
def test_with_scalars(cls, shared_flag, scalar):
    # If scalar is 0-dim np.array, create a copy for convinience. Otherwise, it will be
    # shared by all tests.
    # If scalar is np.number or native int/float, create 0-dim scalar array from it.
    _scalar = np.copy(scalar) if isinstance(scalar, np.ndarray) else np.array(scalar)
    ov_object = cls(array=_scalar, shared_memory=shared_flag)
    if shared_flag is True:
        assert np.shares_memory(_scalar, ov_object.data)
        _scalar[()] = 6
        assert ov_object.data == 6
    else:
        assert not (np.shares_memory(_scalar, ov_object.data))
