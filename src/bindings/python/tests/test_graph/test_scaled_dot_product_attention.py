# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("query", "key", "value", "attention_mask", "scale", "causal", "dtype", "bool_attention"),
    [
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], None, None, None, np.float64, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [1, 3, 7, 10], True, True, np.float32, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [7, 10], False, False, np.float32, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], None, True, False, np.float16, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [1, 3, 7, 10], True, False, np.float32, True),
    ],
)
def test_scaled_dot_product_attention(query, key, value, attention_mask, scale, causal, dtype, bool_attention):
    kwargs = {
        "query": ops.parameter(query, dtype),
        "key": ops.parameter(key, dtype),
        "value": ops.parameter(value, dtype),
    }
    if attention_mask is not None:
        if bool_attention:
            attention_dtype = np.bool_
        else:
            attention_dtype = dtype
        kwargs["attention_mask"] = ops.parameter(attention_mask, attention_dtype)

    if scale is not None:
        kwargs["scale"] = ops.parameter([], dtype)

    if causal is not None:
        kwargs["causal"] = causal

    op = ops.scaled_dot_product_attention(**kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "ScaledDotProductAttention"
    out_shape = query[:-1] + value[-1:]
    assert op.get_output_partial_shape(0) == PartialShape(out_shape)
    assert op.get_output_element_type(0) == Type(dtype)


@pytest.mark.parametrize(
    ("query", "key", "value", "attention_mask", "scale", "causal", "dtype", "bool_attention"),
    [
        ([1, 7, 10], [1, 10, 10], [1, 10, 5], None, None, None, np.float64, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [1, 3, 7, 10], True, True, np.float32, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [7, 10], False, False, np.float32, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], None, True, False, np.float16, False),
        ([3, 3, 7, 10], [3, 3, 10, 10], [3, 3, 10, 5], [1, 3, 7, 10], True, False, np.float32, True),
    ],
)
def test_scaled_dot_product_attention_const(query, key, value, attention_mask, scale, causal, dtype, bool_attention):
    kwargs = {
        "query": ops.constant(np.random.random(query).astype(dtype)),
        "key": ops.constant(np.random.random(key).astype(dtype)),
        "value": ops.constant(np.random.random(value).astype(dtype)),
    }
    if attention_mask is not None:
        if bool_attention:
            attention_dtype = np.bool_
        else:
            attention_dtype = dtype
        kwargs["attention_mask"] = ops.constant(np.random.random(attention_mask).astype(attention_dtype))

    if scale is not None:
        kwargs["scale"] = ops.constant(np.array(np.random.random()).astype(dtype))

    if causal is not None:
        kwargs["causal"] = causal

    op = ops.scaled_dot_product_attention(**kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "ScaledDotProductAttention"
    out_shape = query[:-1] + value[-1:]
    assert op.get_output_partial_shape(0) == PartialShape(out_shape)
    assert op.get_output_element_type(0) == Type(dtype)
