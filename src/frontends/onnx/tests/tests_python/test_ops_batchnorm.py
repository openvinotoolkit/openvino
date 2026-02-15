# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx

from tests.tests_python.utils import run_node


def make_batch_norm_node(**node_attributes):
    return onnx.helper.make_node(
        "BatchNormalization", inputs=["X", "scale", "B", "mean", "var"], outputs=["Y"], **node_attributes,
    )


def test_batch_norm_test_node():
    data = np.arange(48).reshape((1, 3, 4, 4)).astype(np.float32)
    scale = np.ones((3,)).astype(np.float32)  # Gamma
    bias = np.zeros((3,)).astype(np.float32)  # Beta
    mean = np.mean(data, axis=(0, 2, 3))
    variance = np.var(data, axis=(0, 2, 3))

    expected_output = np.array(
        [
            [
                [
                    [-1.62694025, -1.41001487, -1.19308949, -0.97616416],
                    [-0.75923878, -0.54231346, -0.32538807, -0.10846269],
                    [0.10846269, 0.32538807, 0.54231334, 0.75923872],
                    [0.9761641, 1.19308949, 1.41001487, 1.62694025],
                ],
                [
                    [-1.62694049, -1.41001511, -1.19308972, -0.97616434],
                    [-0.7592392, -0.54231358, -0.32538843, -0.10846281],
                    [0.10846233, 0.32538795, 0.5423131, 0.75923872],
                    [0.97616386, 1.19308949, 1.41001463, 1.62694025],
                ],
                [
                    [-1.62694025, -1.41001511, -1.19308949, -0.97616434],
                    [-0.75923872, -0.54231358, -0.32538795, -0.10846233],
                    [0.10846233, 0.32538795, 0.54231358, 0.7592392],
                    [0.97616386, 1.19308949, 1.41001511, 1.62694073],
                ],
            ],
        ],
        dtype=np.float32,
    )

    node = make_batch_norm_node()
    result = run_node(node, [data, scale, bias, mean, variance])[0]
    assert np.allclose(result, expected_output, rtol=1e-04, atol=1e-08)

    scale = np.broadcast_to(0.1, (3,)).astype(np.float32)  # Gamma
    bias = np.broadcast_to(1, (3,)).astype(np.float32)  # Beta

    expected_output = np.array(
        [
            [
                [
                    [0.83730596, 0.85899848, 0.88069105, 0.90238357],
                    [0.92407608, 0.94576865, 0.96746117, 0.98915374],
                    [1.01084626, 1.03253877, 1.05423129, 1.07592392],
                    [1.09761643, 1.11930895, 1.14100146, 1.16269398],
                ],
                [
                    [0.83730596, 0.85899854, 0.88069105, 0.90238357],
                    [0.92407608, 0.94576865, 0.96746117, 0.98915374],
                    [1.01084626, 1.03253877, 1.05423141, 1.07592392],
                    [1.09761643, 1.11930895, 1.14100146, 1.16269398],
                ],
                [
                    [0.83730596, 0.85899848, 0.88069105, 0.90238357],
                    [0.92407614, 0.94576865, 0.96746117, 0.98915374],
                    [1.01084626, 1.03253877, 1.05423141, 1.07592392],
                    [1.09761643, 1.11930895, 1.14100146, 1.16269398],
                ],
            ],
        ],
        dtype=np.float32,
    )

    node = make_batch_norm_node()
    result = run_node(node, [data, scale, bias, mean, variance])[0]
    assert np.allclose(result, expected_output, rtol=1e-04, atol=1e-08)
