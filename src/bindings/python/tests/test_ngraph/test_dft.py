# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset9 as ov
from openvino.runtime import Shape
import numpy as np
from tests.runtime import get_runtime


def build_fft_input_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


def test_dft_1d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                            axis=2).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.00001)


def test_dft_2d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([1, 2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             axes=[1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.000062)


def test_dft_3d():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             axes=[0, 1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.0002)


def test_dft_1d_signal_size():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([-2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([20], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), n=20,
                            axis=-2).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.00001)


def test_dft_2d_signal_size_1():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                             axes=[0, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.000062)


def test_dft_2d_signal_size_2():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fft2(np.squeeze(input_data.view(dtype=np.complex64), axis=-1), s=[4, 5],
                             axes=[1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.000062)


def test_dft_3d_signal_size():
    runtime = get_runtime()
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array([0, 1, 2], dtype=np.int64))
    input_signal_size = ov.constant(np.array([4, 5, 16], dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    computation = runtime.computation(dft_node)
    dft_results = computation()
    np_results = np.fft.fftn(np.squeeze(input_data.view(dtype=np.complex64), axis=-1),
                             s=[4, 5, 16], axes=[0, 1, 2]).astype(np.complex64)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    assert np.allclose(dft_results, expected_results, atol=0.0002)



def test_rdft_1d():
    runtime = get_runtime()
    n = 50
    shape = [n]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array([0], dtype=np.int64))

    node = ov.rdft(param, input_axes)
    computation = runtime.computation(node, param)
    actual = computation(data)
    np_results = np.fft.rfft(data)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)


def test_irdft_1d():
    runtime = get_runtime()
    signal_size = 50
    shape = [signal_size // 2 + 1, 2]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array([0], dtype=np.int64))
    node = ov.irdft(param, input_axes, ov.constant(np.array([signal_size], dtype=np.int64)))
    computation = runtime.computation(node, param)
    actual = computation(data)
    expected_results = np.fft.irfft(data[:, 0] + 1j * data[:, 1], signal_size)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)


def test_rdft_2d():
    runtime = get_runtime()
    shape = [100, 128]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [0, 1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.rdft(param, input_axes)
    computation = runtime.computation(node, param)
    actual = computation(data)
    np_results = np.fft.rfftn(data, axes=axes)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0007)


def test_irdft_2d():
    runtime = get_runtime()
    axes = [0, 1]
    input_shape = [100, 65, 2]
    signal_size = [100, 128]
    data = np.random.uniform(0, 1, input_shape).astype(np.float32)
    param = ov.parameter(Shape(input_shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    computation = runtime.computation(node, param)
    actual = computation(data)
    expected_results = np.fft.irfftn(data[:, :, 0] + 1j * data[:, :, 1], signal_size, axes=axes)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)
