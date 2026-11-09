# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest

from tests.tests_python.utils import run_node


def test_random_uniform():
    low = 90.0
    high = 100.0

    node = onnx.helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["y"],
        high=high,
        low=low,
        seed=10.0,
        shape=(30, 30),
    )

    result = run_node(node, [])[0]

    assert result.shape == (30, 30)
    assert len(np.unique(result)) == 900
    assert np.max(result) < high
    assert np.min(result) > low
    assert np.isclose(np.mean(result), np.mean(np.array([low, high])), rtol=0.5)


def test_random_normal():
    mean = 100.0
    scale = 10.0

    node = onnx.helper.make_node(
        "RandomNormal",
        inputs=[],
        outputs=["y"],
        mean=mean,
        scale=scale,
        seed=10.0,
        shape=(30, 30),
    )

    result = run_node(node, [])[0]

    assert result.shape == (30, 30)
    assert len(np.unique(result)) == 900
    assert np.allclose(np.mean(result), mean, rtol=0.05)
    assert np.allclose(np.std(result), scale, rtol=0.05)


@pytest.mark.parametrize("opset_version", [15, 22])
@pytest.mark.parametrize(
    ("input_dtype", "dtype_attr", "expected_dtype"),
    [
        (np.float32, None, np.float32),
        (np.float64, None, np.float64),
        (np.float16, None, np.float16),
        (np.float32, onnx.TensorProto.INT64, np.int64),
        (np.float32, onnx.TensorProto.BOOL, np.bool_),
    ],
)
def test_bernoulli_degenerate_probabilities(opset_version, input_dtype, dtype_attr, expected_dtype):
    probabilities = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=input_dtype)
    attributes = {} if dtype_attr is None else {"dtype": dtype_attr}
    node = onnx.helper.make_node("Bernoulli", inputs=["x"], outputs=["y"], **attributes)

    result = run_node(
        node,
        [probabilities],
        outputs_info=[(np.dtype(expected_dtype), probabilities.shape)],
        opset_version=opset_version,
    )[0]

    assert result.dtype == expected_dtype
    assert np.array_equal(result, probabilities.astype(expected_dtype))


def test_bernoulli_seed():
    probability = 0.3
    probabilities = np.full((100, 100), probability, dtype=np.float32)
    node = onnx.helper.make_node("Bernoulli", inputs=["x"], outputs=["y"], seed=7.0)

    result = run_node(node, [probabilities], opset_version=15)[0]
    repeated = run_node(node, [probabilities], opset_version=15)[0]

    assert result.shape == probabilities.shape
    assert set(np.unique(result)) <= {0.0, 1.0}
    assert np.array_equal(result, repeated)
    assert np.isclose(np.mean(result), probability, atol=0.03)
