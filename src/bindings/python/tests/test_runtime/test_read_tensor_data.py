# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import openvino as ov


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    assert path.exists()


@pytest.mark.parametrize(
    ("dtype", "ov_type"),
    [
        (np.float32, ov.Type.f32),
        (np.float64, ov.Type.f64),
        (np.int8, ov.Type.i8),
        (np.int16, ov.Type.i16),
        (np.int32, ov.Type.i32),
        (np.int64, ov.Type.i64),
        (np.uint8, ov.Type.u8),
        (np.uint16, ov.Type.u16),
        (np.uint32, ov.Type.u32),
        (np.uint64, ov.Type.u64),
    ],
)
@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_typed(tmp_path: Path, dtype: np.dtype, ov_type: ov.Type, mmap: bool) -> None:
    shape = (10, 20, 3, 2)
    data = np.random.randint(0, 100, size=np.prod(shape)).astype(dtype, copy=False).reshape(shape)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    tensor = ov.read_tensor_data(path, element_type=ov_type, shape=ov.PartialShape(list(shape)), mmap=mmap)
    assert tensor.get_shape() == list(shape)
    assert tensor.get_element_type() == ov_type
    assert np.array_equal(tensor.data, data)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_string_tensor_throws(tmp_path: Path, mmap: bool) -> None:
    path = tmp_path / "tensor.bin"
    _write_bytes(path, b"abc")
    with pytest.raises(RuntimeError):
        ov.read_tensor_data(path, element_type=ov.Type.string, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_with_offset(tmp_path: Path, mmap: bool) -> None:
    shape = (1, 2, 3, 4)
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32).reshape(shape)
    dummy = np.array([0.0], dtype=np.float32)

    path = tmp_path / "tensor.bin"
    _write_bytes(path, dummy.tobytes() + data.tobytes())

    tensor = ov.read_tensor_data(
        path,
        element_type=ov.Type.f32,
        shape=ov.PartialShape(list(shape)),
        offset_in_bytes=dummy.nbytes,
        mmap=mmap,
    )
    assert tensor.get_shape() == list(shape)
    assert np.array_equal(tensor.data, data)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_small_file_throws(tmp_path: Path, mmap: bool) -> None:
    shape = (1, 2, 3, 4)
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    too_big_shape = ov.PartialShape([10, 2, 3, 4])
    with pytest.raises(RuntimeError):
        ov.read_tensor_data(path, element_type=ov.Type.f32, shape=too_big_shape, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_too_big_offset_throws(tmp_path: Path, mmap: bool) -> None:
    shape = (1, 2, 3, 4)
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    file_size = path.stat().st_size
    assert file_size == data.nbytes

    with pytest.raises(RuntimeError):
        ov.read_tensor_data(
            path,
            element_type=ov.Type.f32,
            shape=ov.PartialShape(list(shape)),
            offset_in_bytes=1,
            mmap=mmap)

    with pytest.raises(RuntimeError):
        ov.read_tensor_data(
            path,
            element_type=ov.Type.f32,
            shape=ov.PartialShape(list(shape)),
            offset_in_bytes=file_size,
            mmap=mmap,
        )

    with pytest.raises(RuntimeError):
        ov.read_tensor_data(
            path,
            element_type=ov.Type.f32,
            shape=ov.PartialShape(list(shape)),
            offset_in_bytes=file_size + 1,
            mmap=mmap,
        )


def test_read_tensor_data_default_all_args(tmp_path: Path) -> None:
    """Test main use case: only path specified, all other args use defaults."""
    data = np.arange(24, dtype=np.uint8)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    # Main use case - only specify path
    tensor = ov.read_tensor_data(path)

    assert isinstance(tensor, ov.Tensor)
    assert tensor.get_element_type() == ov.Type.u8  # default element type
    assert tensor.get_shape() == [data.size]  # dynamic shape inferred from file
    assert not tensor.data.flags.writeable  # read-only
    assert np.array_equal(tensor.data, data)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_dynamic_shape(tmp_path: Path, mmap: bool) -> None:
    shape = (1, 2, 3, 4)
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    tensor = ov.read_tensor_data(path, element_type=ov.Type.f32, shape=ov.PartialShape.dynamic(1), mmap=mmap)
    assert tensor.get_shape() == [data.size]
    assert np.array_equal(tensor.data, data)

    # default element type is u8 and default shape is dynamic(1)
    tensor_u8 = ov.read_tensor_data(path, mmap=mmap)
    expected_u8 = np.fromfile(path, dtype=np.uint8)
    assert tensor_u8.get_shape() == [expected_u8.size]
    assert tensor_u8.get_element_type() == ov.Type.u8
    assert np.array_equal(tensor_u8.data, expected_u8)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_1_dynamic_dimension(tmp_path: Path, mmap: bool) -> None:
    # Last dimension is inferred from file size
    shape = [1, 2, 3, 4]
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    shape_with_dynamic_last = ov.PartialShape([1, 2, 3, -1])
    tensor = ov.read_tensor_data(path, element_type=ov.Type.f32, shape=shape_with_dynamic_last, mmap=mmap)
    assert tensor.get_shape()[-1] == shape[-1]
    assert np.array_equal(tensor.data, data.reshape(shape))


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_wrong_dynamic_shape_throws(tmp_path: Path, mmap: bool) -> None:
    shape = [1, 2, 3, 4]
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    wrong_shape = ov.PartialShape([1, 2, 100, -1])
    with pytest.raises(RuntimeError):
        ov.read_tensor_data(path, element_type=ov.Type.f32, shape=wrong_shape, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_type_doesnt_fit_file_size(tmp_path: Path, mmap: bool) -> None:
    path = tmp_path / "tensor.bin"
    # 3 bytes: not divisible by sizeof(float)
    _write_bytes(path, b"abc")
    with pytest.raises(RuntimeError):
        ov.read_tensor_data(path, element_type=ov.Type.f32, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_null_shape_throws(tmp_path: Path, mmap: bool) -> None:
    shape = [1, 2, 3, 4]
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    # One null dimension (0) and one dynamic dimension
    null_shape = ov.PartialShape([0, ov.Dimension.dynamic(), 3, 4])
    with pytest.raises(RuntimeError):
        ov.read_tensor_data(path, element_type=ov.Type.f32, shape=null_shape, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_read_tensor_data_returns_readonly_array(tmp_path: Path, mmap: bool) -> None:
    """Test that tensors from read_tensor_data have read-only numpy arrays."""
    shape = (2, 3, 4)
    data = np.random.uniform(0.0, 1.0, size=np.prod(shape)).astype(np.float32).reshape(shape)
    path = tmp_path / "tensor.bin"
    data.tofile(path)

    tensor = ov.read_tensor_data(path, element_type=ov.Type.f32, shape=ov.PartialShape(list(shape)), mmap=mmap)

    # Verify the numpy array is read-only
    assert not tensor.data.flags.writeable

    # Verify attempting to write raises an error
    with pytest.raises(ValueError, match="read-only"):
        tensor.data[0, 0, 0] = 999.0
