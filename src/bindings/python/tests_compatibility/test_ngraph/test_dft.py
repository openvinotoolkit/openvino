# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph.impl import Type
import ngraph as ng
import numpy as np


def build_fft_input_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


def test_dft_1d():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([2], dtype=np.int64))
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                            axis=2).astype(np.complex64)

    dft_node = ng.dft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(np.stack((np_results.real, np_results.imag), axis=-1).shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_2d():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([1, 2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [2, 10, 10, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_3d():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 1, 2], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [2, 10, 10, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_1d_signal_size():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([-2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([20], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [2, 20, 10, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_2d_signal_size_1():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [4, 10, 5, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_2d_signal_size_2():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([1, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [2, 4, 5, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


def test_dft_3d_signal_size():
    input_data = build_fft_input_data()
    input_tensor = ng.constant(input_data)
    input_axes = ng.constant(np.array([0, 1, 2], dtype=np.int64))
    input_signal_size = ng.constant(np.array([4, 5, 16], dtype=np.int64))

    dft_node = ng.dft(input_tensor, input_axes, input_signal_size)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [4, 5, 16, 2]
    assert dft_node.get_output_element_type(0) == Type.f32
