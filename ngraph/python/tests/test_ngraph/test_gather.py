# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np

from tests.test_ngraph.util import run_op_node


def test_gather():
    input_data = np.array(
        [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2], np.float32
    ).reshape((3, 3))
    input_indices = np.array([0, 2], np.int32).reshape(1, 2)
    input_axis = np.array([1], np.int32)

    expected = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float32).reshape(
        (3, 1, 2)
    )

    result = run_op_node([input_data], ng.gather, input_indices, input_axis)
    assert np.allclose(result, expected)


def test_gather_with_scalar_axis():
    input_data = np.array(
        [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2], np.float32
    ).reshape((3, 3))
    input_indices = np.array([0, 2], np.int32).reshape(1, 2)
    input_axis = np.array(1, np.int32)

    expected = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float32).reshape(
        (3, 1, 2)
    )

    result = run_op_node([input_data], ng.gather, input_indices, input_axis)
    assert np.allclose(result, expected)


def test_gather_batch_dims_1():

    input_data = np.array([[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10]], np.float32)

    input_indices = np.array([[0, 0, 4],
                              [4, 0, 0]], np.int32)
    input_axis = np.array([1], np.int32)
    batch_dims = 1

    expected = np.array([[1, 1, 5],
                         [10, 6, 6]], np.float32)

    result = run_op_node([input_data], ng.gather, input_indices, input_axis, batch_dims)
    assert np.allclose(result, expected)
