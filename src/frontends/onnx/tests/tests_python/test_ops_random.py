# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnx.mapping

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
