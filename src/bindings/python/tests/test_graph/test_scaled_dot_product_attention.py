# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset13 as ops
from openvino.runtime import PartialShape, Type


@pytest.mark.parametrize(
    ["query", "key", "value", "attention_mask", "scale", "causal", "dtype"],
    [
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], None, None, None, np.float32),
        ([1, 3, 1, 7, 10], [1, 3, 1, 10, 10], [1, 3, 1, 10, 5], None, [1], False, np.float32),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [10, 7, 10], None, False, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [10, 7, 10], None, True, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [0], None, False, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], None, None, False, np.float64),
    ],
)
def test_scaled_dot_product_attention(query, key, value, attention_mask, scale, causal, dtype):
    kwargs = {
        "query": ops.parameter(query, dtype),  # [N, ..., L, E]
        "key": ops.parameter(key, dtype),  # [N, ..., S, E]
        "value": ops.parameter(value, dtype),  # [N, ..., S, EV]
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = ops.parameter(attention_mask, dtype)

    if scale is not None:
        kwargs["scale"] = ops.parameter(scale, dtype)

    if causal is not None:
        kwargs["causal"] = causal

    op = ops.scaled_dot_product_attention(**kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "ScaledDotProductAttention"
    out_shape = query[:-1] + value[-1:]
    assert op.get_output_partial_shape(0) == PartialShape(out_shape)  # [N, ..., L, EV]
    assert op.get_output_element_type(0) == Type(dtype)


@pytest.mark.parametrize(
    ["query", "key", "value", "attention_mask", "scale", "causal", "dtype"],
    [
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], None, None, None, np.float32),
        ([1, 3, 1, 7, 10], [1, 3, 1, 10, 10], [1, 3, 1, 10, 5], None, [1], False, np.float32),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [10, 7, 10], None, False, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [10, 7, 10], None, True, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], [0], None, False, np.float64),
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], None, None, False, np.float64),
    ],
)
def test_scaled_dot_product_attention_const(query, key, value, attention_mask, scale, causal, dtype):
    kwargs = {
        "query": ops.constant(np.random.random(query).astype(dtype)),  # [N, ..., L, E]
        "key": ops.constant(np.random.random(key).astype(dtype)),  # [N, ..., S, E]
        "value": ops.constant(np.random.random(value).astype(dtype)),  # [N, ..., S, EV]
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = ops.constant(np.random.random(attention_mask).astype(dtype))

    if scale is not None:
        kwargs["scale"] = ops.constant(np.random.random(scale).astype(dtype))

    if causal is not None:
        kwargs["causal"] = causal

    op = ops.scaled_dot_product_attention(**kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "ScaledDotProductAttention"
    out_shape = query[:-1] + value[-1:]
    assert op.get_output_partial_shape(0) == PartialShape(out_shape)  # [N, ..., L, EV]
    assert op.get_output_element_type(0) == Type(dtype)
