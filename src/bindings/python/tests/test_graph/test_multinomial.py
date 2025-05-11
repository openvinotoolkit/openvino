# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("probs_shape", "num_samples_shape", "convert_type", "with_replacement", "log_probs", "global_seed", "op_seed", "expected_out_shape"),
    [
        ([4, 16], [], "i32", False, True, 7461, 1546, PartialShape([4, -1])),
        ([1, 8], [1], "i64", True, False, 0, 0, PartialShape([1, -1])),
    ],
)
@pytest.mark.parametrize("op_name", ["multinomial", "multinomialOpset13"])
def test_multinomial_param_inputs(probs_shape, num_samples_shape, convert_type,
                                  with_replacement, log_probs, global_seed, op_seed, expected_out_shape, op_name):
    probs = ops.parameter(probs_shape, dtype=np.float32)
    num_samples = ops.parameter(num_samples_shape, dtype=np.int32)

    op = ops.multinomial(probs, num_samples,
                         convert_type=convert_type,
                         with_replacement=with_replacement,
                         log_probs=log_probs,
                         global_seed=global_seed,
                         op_seed=op_seed,
                         name=op_name)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "Multinomial"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.i32 if convert_type == "i32" else Type.i64
    assert op.get_output_partial_shape(0) == expected_out_shape


@pytest.mark.parametrize(
    ("probs_array", "num_samples_val", "convert_type", "with_replacement", "log_probs", "global_seed", "op_seed", "expected_out_shape"),
    [
        (np.array([[0.7, 0.3, 0.6, 0.5]]), 3, "i32", False, True, 111, 222, PartialShape([1, 3])),
        (np.array([[0.7, 0.3], [0.6, 0.5]]), 2, "i64", True, False, 111, 222, PartialShape([2, 2])),
    ],
)
@pytest.mark.parametrize("op_name", ["multinomial", "multinomialOpset13"])
def test_multinomial_const_inputs(probs_array, num_samples_val, convert_type,
                                  with_replacement, log_probs, global_seed, op_seed, expected_out_shape, op_name):
    probs = ops.constant(probs_array, dtype=np.float32)
    num_samples = ops.constant(num_samples_val, dtype=np.int32)

    op = ops.multinomial(probs, num_samples,
                         convert_type=convert_type,
                         with_replacement=with_replacement,
                         log_probs=log_probs,
                         global_seed=global_seed,
                         op_seed=op_seed,
                         name=op_name)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Multinomial"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.i32 if convert_type == "i32" else Type.i64
    assert op.get_output_partial_shape(0) == expected_out_shape


@pytest.mark.parametrize(
    ("probs_shape", "num_samples_shape", "convert_type", "with_replacement", "log_probs", "expected_out_shape"),
    [
        ([1, 10], [1], "i32", True, True, PartialShape([1, -1])),
        ([2, 16], [], "i64", False, False, PartialShape([2, -1])),
    ],
)
@pytest.mark.parametrize("op_name", ["multinomial", "multinomialOpset13"])
def test_multinomial_default_attrs(probs_shape, num_samples_shape, convert_type,
                                   with_replacement, log_probs, expected_out_shape, op_name):
    probs = ops.parameter(probs_shape, dtype=np.float32)
    num_samples = ops.parameter(num_samples_shape, dtype=np.int32)

    op = ops.multinomial(probs, num_samples,
                         convert_type=convert_type,
                         with_replacement=with_replacement,
                         log_probs=log_probs,
                         name=op_name)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Multinomial"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.i32 if convert_type == "i32" else Type.i64
    assert op.get_output_partial_shape(0) == expected_out_shape
