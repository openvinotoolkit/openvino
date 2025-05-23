# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset10 as ov
import numpy as np
import pytest

from openvino.utils.types import get_element_type_str
from openvino.utils.types import get_element_type


@pytest.mark.parametrize(
    ("num_rows", "num_columns", "diagonal_index", "out_type"),
    [
        pytest.param(2, 5, 0, np.float32),
        pytest.param(5, 3, 2, np.int64),
        pytest.param(3, 3, -1, np.float16),
        pytest.param(5, 5, -10, np.float32),
    ],
)
def test_eye_rectangle(num_rows, num_columns, diagonal_index, out_type):
    num_rows_array = np.array([num_rows], np.int32)
    num_columns_array = np.array([num_columns], np.int32)
    diagonal_index_array = np.array([diagonal_index], np.int32)
    num_rows_tensor = ov.constant(num_rows_array)
    num_columns_tensor = ov.constant(num_columns_array)
    diagonal_index_tensor = ov.constant(diagonal_index_array)

    # Create with param names
    eye_node = ov.eye(num_rows=num_rows_tensor,
                      num_columns=num_columns_tensor,
                      diagonal_index=diagonal_index_tensor,
                      output_type=get_element_type_str(out_type))

    # Create with default orded
    eye_node = ov.eye(num_rows_tensor,
                      num_columns_tensor,
                      diagonal_index_tensor,
                      get_element_type_str(out_type))

    expected_results = np.eye(num_rows, M=num_columns, k=diagonal_index, dtype=np.float32)

    assert eye_node.get_type_name() == "Eye"
    assert eye_node.get_output_size() == 1
    assert eye_node.get_output_element_type(0) == get_element_type(out_type)
    assert tuple(eye_node.get_output_shape(0)) == expected_results.shape


@pytest.mark.parametrize(
    ("num_rows", "num_columns", "diagonal_index", "batch_shape", "out_type"),
    [
        pytest.param(2, 5, 0, [1], np.float32),
        pytest.param(5, 3, 2, [2, 2], np.int64),
        pytest.param(3, 3, -1, [1, 3, 2], np.float16),
        pytest.param(5, 5, -10, [1, 1], np.float32),
    ],
)
def test_eye_batch_shape(num_rows, num_columns, diagonal_index, batch_shape, out_type):
    num_rows_array = np.array([num_rows], np.int32)
    num_columns_array = np.array([num_columns], np.int32)
    diagonal_index_array = np.array([diagonal_index], np.int32)
    batch_shape_array = np.array(batch_shape, np.int32)
    num_rows_tensor = ov.constant(num_rows_array)
    num_columns_tensor = ov.constant(num_columns_array)
    diagonal_index_tensor = ov.constant(diagonal_index_array)
    batch_shape_tensor = ov.constant(batch_shape_array)

    # Create with param names
    eye_node = ov.eye(num_rows=num_rows_tensor,
                      num_columns=num_columns_tensor,
                      diagonal_index=diagonal_index_tensor,
                      batch_shape=batch_shape_tensor,
                      output_type=get_element_type_str(out_type))

    # Create with default orded
    eye_node = ov.eye(num_rows_tensor,
                      num_columns_tensor,
                      diagonal_index_tensor,
                      get_element_type_str(out_type),
                      batch_shape_tensor)

    output_shape = [*batch_shape, 1, 1]
    one_matrix = np.eye(num_rows, M=num_columns, k=diagonal_index, dtype=np.float32)
    expected_results = np.tile(one_matrix, output_shape)

    assert eye_node.get_type_name() == "Eye"
    assert eye_node.get_output_size() == 1
    assert eye_node.get_output_element_type(0) == get_element_type(out_type)
    assert tuple(eye_node.get_output_shape(0)) == expected_results.shape
