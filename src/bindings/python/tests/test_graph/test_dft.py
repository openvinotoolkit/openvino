# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
import openvino.opset10 as ov
import numpy as np
import pytest


def build_fft_input_data():
    np.random.seed(202104)
    return np.random.uniform(0, 1, (2, 10, 10, 2)).astype(np.float32)


@pytest.mark.parametrize("dims", [[2], [1, 2], [0, 1, 2]])
def test_dft_dims(dims):
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array(dims, dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == [2, 10, 10, 2]
    assert dft_node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("dims", "signal_size", "expected_shape"), [
    ([-2], [20], [2, 20, 10, 2]),
    ([0, 2], [4, 5], [4, 10, 5, 2]),
    ([1, 2], [4, 5], [2, 4, 5, 2]),
    ([0, 1, 2], [4, 5, 16], [4, 5, 16, 2]),
])
def test_dft_signal_size(dims, signal_size, expected_shape):
    input_data = build_fft_input_data()
    input_tensor = ov.constant(input_data)
    input_axes = ov.constant(np.array(dims, dtype=np.int64))
    input_signal_size = ov.constant(np.array(signal_size, dtype=np.int64))

    dft_node = ov.dft(input_tensor, input_axes, input_signal_size)
    assert dft_node.get_type_name() == "DFT"
    assert dft_node.get_output_size() == 1
    assert list(dft_node.get_output_shape(0)) == expected_shape
    assert dft_node.get_output_element_type(0) == Type.f32
