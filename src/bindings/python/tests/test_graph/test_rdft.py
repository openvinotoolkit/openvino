# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset10 as ov
from openvino import Shape, Type
import numpy as np
import pytest


np.random.seed(0)


@pytest.mark.parametrize(("shape", "axes", "expected_shape"), [
    ([50], [0], [26, 2]),
    ([100, 128], [0, 1], [100, 65, 2]),
    ([1, 192, 36, 64], [-2, -1], [1, 192, 36, 33, 2]),
])
def test_rdft(shape, axes, expected_shape):
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))

    node = ov.rdft(param, input_axes)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("shape", "axes", "expected_shape"), [
    ([100, 65, 2], [0, 1], [100, 128]),
    ([1, 192, 36, 33, 2], [-2, -1], [1, 192, 36, 64]),
])
def test_irdft(shape, axes, expected_shape):
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.irdft(param, input_axes)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("shape", "axes", "expected_shape", "signal_size"), [
    ([26, 2], [0], [50], [50]),
    ([100, 65, 2], [0, 1], [100, 65], [100, 65]),
    ([1, 192, 36, 33, 2], [-2, -1], [1, 192, 36, 64], [36, 64]),
])
def test_irdft_signal_size(shape, axes, expected_shape, signal_size):
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("shape", "axes", "expected_shape", "signal_size"), [
    ([100, 128], [0, 1], [30, 21, 2], [30, 40]),
    ([1, 192, 36, 64], [-2, -1], [1, 192, 36, 33, 2], [36, 64]),
])
def test_rdft_signal_size(shape, axes, expected_shape, signal_size):
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes_node = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.rdft(param, axes_node, signal_size_node)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32
