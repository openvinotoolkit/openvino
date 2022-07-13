# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset9 as ov
from openvino.runtime import Shape
import numpy as np
from tests.runtime import get_runtime

np.random.seed(0)

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


def test_rdft_2d_signal_size():
    runtime = get_runtime()
    shape = [100, 128]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [0, 1]
    signal_size = [30, 40]
    axes_node = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.rdft(param, axes_node, signal_size)
    computation = runtime.computation(node, param)
    actual = computation(data)
    np_results = np.fft.rfftn(data, s=signal_size, axes=axes)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0007)


def test_irdft_2d():
    runtime = get_runtime()
    axes = [0, 1]
    input_shape = [100, 65, 2]
    data = np.random.uniform(0, 1, input_shape).astype(np.float32)
    param = ov.parameter(Shape(input_shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.irdft(param, input_axes)
    computation = runtime.computation(node, param)
    actual = computation(data)
    expected_results = np.fft.irfftn(data[:, :, 0] + 1j * data[:, :, 1], axes=axes)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)


def test_irdft_2d_signal_size():
    runtime = get_runtime()
    axes = [0, 1]
    input_shape = [100, 65, 2]
    signal_size = [100, 65]
    data = np.random.uniform(0, 1, input_shape).astype(np.float32)
    param = ov.parameter(Shape(input_shape), name="input", dtype=np.float32)
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    computation = runtime.computation(node, param)
    actual = computation(data)
    expected_results = np.fft.irfftn(data[:, :, 0] + 1j * data[:, :, 1], s=signal_size, axes=axes)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)


def test_rdft_4d():
    runtime = get_runtime()
    shape = [1, 192, 33, 64]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    node = ov.rdft(param, input_axes)
    computation = runtime.computation(node, param)
    actual = computation(data)
    np_results = np.fft.rfftn(data, axes=axes)
    expected_results = np.stack((np_results.real, np_results.imag), axis=-1)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0007)


def test_irdft_4d():
    runtime = get_runtime()
    shape = [1, 192, 33, 33, 2]
    signal_size = [33, 64]
    data = np.random.uniform(0, 1, shape).astype(np.float32)
    param = ov.parameter(Shape(shape), name="input", dtype=np.float32)
    axes = [-2, -1]
    input_axes = ov.constant(np.array(axes, dtype=np.int64))
    signal_size_node = ov.constant(np.array(signal_size, dtype=np.int64))
    node = ov.irdft(param, input_axes, signal_size_node)
    computation = runtime.computation(node, param)
    actual = computation(data)
    np_results = np.fft.rfftn(data, axes=axes)
    expected_results = np.fft.irfftn(data[:, :, :, :, 0] + 1j * data[:, :, :, :, 1], signal_size, axes=axes)
    np.testing.assert_allclose(expected_results, actual[0], atol=0.0001)
