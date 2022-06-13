# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset8 as ov
import numpy as np
from tests.runtime import get_runtime


def test_roll():
    runtime = get_runtime()
    input_vals = np.reshape(np.arange(10), (2, 5))
    input_tensor = ov.constant(input_vals)
    input_shift = ov.constant(np.array([-10, 7], dtype=np.int32))
    input_axes = ov.constant(np.array([-1, 0], dtype=np.int32))

    roll_node = ov.roll(input_tensor, input_shift, input_axes)
    computation = runtime.computation(roll_node)
    roll_results = computation()
    expected_results = np.roll(input_vals, shift=(-10, 7), axis=(-1, 0))

    assert np.allclose(roll_results, expected_results)
