# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
import openvino.opset8 as ov
import numpy as np
import pytest


def get_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


def test_idft_1d():
    expected_results = get_data()
    complex_input_data = np.fft.fft(np.squeeze(expected_results.view(dtype=np.complex64),
                                    axis=-1), axis=2).astype(np.complex64)
    input_data = np.stack((complex_input_data.real, complex_input_data.imag), axis=-1)
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([2], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("axes"), [
    ([1, 2]),
    ([0, 1, 2]),
])
def test_idft_2d_3d(axes):
    expected_results = get_data()
    complex_input_data = np.fft.fft2(np.squeeze(expected_results.view(dtype=np.complex64), axis=-1),
                                     axes=axes).astype(np.complex64)
    input_data = np.stack((complex_input_data.real, complex_input_data.imag), axis=-1)
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_idft_3d():
    expected_results = get_data()
    complex_input_data = np.fft.fft2(np.squeeze(expected_results.view(dtype=np.complex64), axis=-1),
                                     axes=[0, 1, 2]).astype(np.complex64)
    input_data = np.stack((complex_input_data.real, complex_input_data.imag), axis=-1)
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_idft_1d_signal_size():
    input_data = get_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([-2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([20], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.ifft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), n=20,
                             axis=-2).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_idft_2d_signal_size_1():
    input_data = get_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.ifft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                              axes=[0, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_idft_2d_signal_size_2():
    input_data = get_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.ifft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                              axes=[1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32


def test_idft_3d_signal_size():
    input_data = get_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5, 16], dtype=np.int64))

    dft_node = ov.idft(input_tensor, input_axes, input_signal_size)
    np_results = np.fft.ifftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                              s=[4, 5, 16], axes=[0, 1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert dft_node.get_type_name() == "IDFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == list(expected_results.shape)
    assert dft_node.get_output_element_type(0) == Type.f32
