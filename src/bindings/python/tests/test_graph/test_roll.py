# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset8 as ov
import numpy as np


def test_roll():
    input_vals = np.reshape(np.arange(10), (2, 5))
    input_tensor = ov.constant(input_vals)
    input_shift = ov.constant(np.array([-10, 7], dtype=np.int32))
    input_axes = ov.constant(np.array([-1, 0], dtype=np.int32))

    roll_node = ov.roll(input_tensor, input_shift, input_axes)
    assert roll_node.get_output_size() == 1
    assert roll_node.get_type_name() == "Roll"
    assert list(roll_node.get_output_shape(0)) == [2, 5]
