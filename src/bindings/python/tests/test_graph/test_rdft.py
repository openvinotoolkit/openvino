# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset9 as ov
from openvino.runtime import Shape, Type
import numpy as np


np.random.seed(0)


def test_rdft_1d():
    input_size = 50
    shape = [input_size]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array([0], dtype=np.int64))

    node = ov.rdft(param, input_axes)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [26, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft_1d():
    signal_size = 50
    shape = [signal_size // 2 + 1, 2]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array([0], dtype=np.int64))
    node = ov.irdft(param, input_axes, ov.constant(np.array([signal_size], dtype=np.int64)))
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [50]
    assert node.get_output_element_type(0) == Type.f32


def test_rdft_2d():
    shape = [100, 128]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [0, 1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.rdft(param, input_axes)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [100, 65, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_rdft_2d_signal_size():
    shape = [100, 128]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [0, 1]
    signal_size = [30, 40]
    axes_node = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.rdft(param, axes_node, signal_size_node)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [30, 21, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft_2d():
    axes = [0, 1]
    input_shape = [100, 65, 2]
    param = ov.parameter(Shape(input_shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.irdft(param, input_axes)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [100, 128]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft_2d_signal_size():
    axes = [0, 1]
    input_shape = [100, 65, 2]
    signal_size = [100, 65]
    param = ov.parameter(Shape(input_shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [100, 65]
    assert node.get_output_element_type(0) == Type.f32


def test_rdft_4d():
    shape = [1, 192, 36, 64]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.rdft(param, input_axes)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 192, 36, 33, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_rdft_4d_signal_size():
    shape = [1, 192, 36, 64]
    signal_size = [36, 64]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.rdft(param, input_axes, signal_size_node)
    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 192, 36, 33, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft_4d():
    shape = [1, 192, 36, 33, 2]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.irdft(param, input_axes)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 192, 36, 64]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft_4d_signal_size():
    shape = [1, 192, 36, 33, 2]
    signal_size = [36, 64]
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 192, 36, 64]
    assert node.get_output_element_type(0) == Type.f32
