# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset8 as ov
import numpy as np
import pytest

from openvino.utils.types import get_element_type


def einsum_op_check(input_shapes: list, equation: str, data_type: np.dtype,
                    seed=202104):
    """Test Einsum operation for given input shapes, equation, and data type.

    It generates input data of given shapes and type, receives reference results using numpy,
    and tests OV implementation by matching with reference numpy results.
    :param input_shapes: a list of tuples with shapes
    :param equation: Einsum equation
    :param data_type: a type of input data

    :param seed: a seed for random generation of input data
    """
    np.random.seed(seed)
    num_inputs = len(input_shapes)

    # generate input tensors
    graph_inputs = []
    np_inputs = []
    for i in range(num_inputs):
        input_i = np.random.randint(1, 10 + 1, size=input_shapes[i]).astype(data_type)
        np_inputs.append(input_i)
        graph_inputs.append(ov.parameter(input_i.shape, dtype=data_type))

    expected_result = np.einsum(equation, *np_inputs)
    einsum_model = ov.einsum(graph_inputs, equation)

    # check the output shape and type
    assert einsum_model.get_type_name() == "Einsum"
    assert einsum_model.get_output_size() == 1
    assert list(einsum_model.get_output_shape(0)) == list(expected_result.shape)
    assert einsum_model.get_output_element_type(0) == get_element_type(data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_dot_product(data_type):
    einsum_op_check([5, 5], "i,i->", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_matrix_multiplication(data_type):
    einsum_op_check([(2, 3), (3, 4)], "ab,bc->ac", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_batch_trace(data_type):
    einsum_op_check([(2, 3, 3)], "kii->k", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_diagonal_extraction(data_type):
    einsum_op_check([(6, 5, 5)], "kii->ki", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_transpose(data_type):
    einsum_op_check([(1, 2, 3)], "ijk->kij", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_multiple_multiplication(data_type):
    einsum_op_check([(2, 5), (5, 3, 6), (5, 3)], "ab,bcd,bc->ca", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_simple_ellipsis(data_type):
    einsum_op_check([(5, 3, 4)], "a...->...", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_multiple_ellipsis(data_type):
    einsum_op_check([(3, 5), 1], "a...,...->a...", data_type)


@pytest.mark.parametrize("data_type", [np.float32, np.int32])
def test_broadcasting_ellipsis(data_type):
    einsum_op_check([(9, 1, 4, 3), (3, 11, 7, 1)], "a...b,b...->a...", data_type)
