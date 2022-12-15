# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime.opset8 as ov


def test_onehot():
    param = ov.parameter([3], dtype=np.int32)
    model = ov.one_hot(param, 3, 1, 0, 0)
    assert model.get_output_size() == 1
    assert model.get_type_name() == "OneHot"
    assert list(model.get_output_shape(0)) == [3, 3]


def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1

    node = ov.one_hot(data, depth, on_value, off_value, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "OneHot"
    assert list(node.get_output_shape(0)) == [3, 2]


def test_range():
    start = 5
    stop = 35
    step = 5

    node = ov.range(start, stop, step)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Range"
    assert list(node.get_output_shape(0)) == [6]
