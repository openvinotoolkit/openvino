# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import openvino.runtime.opset12 as ops
from openvino import Type


def test_output_replace(device):
    param = ops.parameter([1, 64], Type.i64)
    param.output(0).get_tensor().set_names({"a", "b"})
    relu = ops.relu(param)
    relu.output(0).get_tensor().set_names({"c", "d"})

    new_relu = ops.relu(param)
    new_relu.output(0).get_tensor().set_names({"f"})

    relu.output(0).replace(new_relu.output(0))

    assert new_relu.output(0).get_tensor().get_names() == {"c", "d", "f"}
