# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset9 as ov
import numpy as np
import pytest
from tests.runtime import get_runtime
from openvino.runtime.utils.types import get_element_type_str


def test_eye_default():
    runtime = get_runtime()
    num_rows = np.array([3], np.int32)
    num_rows_tensor = ov.constant(num_rows)

    eye_node = ov.eye(num_rows_tensor, output_type="f32")
    computation = runtime.computation(eye_node)
    eye_results = computation()
    expected_results = np.eye(3, dtype=np.float32)

    assert np.allclose(eye_results, expected_results)


@pytest.mark.parametrize(
    "num_rows, num_columns, diagonal_index, output_type",
    [
        pytest.param(2, 5, 0, np.float32),
        pytest.param(5, 3, 2, np.int64),
        pytest.param(3, 3, -1, np.float16),
        pytest.param(5, 5, -10, np.float32),
    ],
)
def test_eye_rectangle(num_rows, num_columns, diagonal_index, output_type):
    runtime = get_runtime()
    num_rows_array = np.array([num_rows], np.int32)
    num_columns_array = np.array([num_columns], np.int32)
    diagonal_index_array = np.array([diagonal_index], np.int32)
    num_rows_tensor = ov.constant(num_rows_array)
    num_columns_tensor = ov.constant(num_columns_array)
    diagonal_index_tensor = ov.constant(diagonal_index_array)

    eye_node = ov.eye(num_rows_tensor,
                      num_columns=num_columns_tensor,
                      diagonal_index=diagonal_index_tensor,
                      output_type=get_element_type_str(output_type))
    computation = runtime.computation(eye_node)
    eye_results = computation()
    expected_results = np.eye(num_rows, M=num_columns, k=diagonal_index, dtype=np.float32)
    assert np.allclose(eye_results, expected_results)


@pytest.mark.parametrize(
    "num_rows, num_columns, diagonal_index, batch_shape, output_type",
    [
        pytest.param(2, 5, 0, [1], np.float32),
        pytest.param(5, 3, 2, [2, 2], np.int64),
        pytest.param(3, 3, -1, [1, 3, 2], np.float16),
        pytest.param(5, 5, -10, [1, 1], np.float32),
    ],
)
def test_eye_batch_shape(num_rows, num_columns, diagonal_index, batch_shape, output_type):
    runtime = get_runtime()
    num_rows_array = np.array([num_rows], np.int32)
    num_columns_array = np.array([num_columns], np.int32)
    diagonal_index_array = np.array([diagonal_index], np.int32)
    batch_shape_array = np.array(batch_shape, np.int32)
    num_rows_tensor = ov.constant(num_rows_array)
    num_columns_tensor = ov.constant(num_columns_array)
    diagonal_index_tensor = ov.constant(diagonal_index_array)
    batch_shape_tensor = ov.constant(batch_shape_array)

    eye_node = ov.eye(num_rows_tensor,
                      num_columns=num_columns_tensor,
                      diagonal_index=diagonal_index_tensor,
                      batch_shape=batch_shape_tensor,
                      output_type=get_element_type_str(output_type))
    computation = runtime.computation(eye_node)
    eye_results = computation()

    output_shape = [*batch_shape, 1, 1]
    one_matrix = np.eye(num_rows, M=num_columns, k=diagonal_index, dtype=np.float32)
    expected_results = np.tile(one_matrix, output_shape)
    assert np.allclose(eye_results, expected_results)
