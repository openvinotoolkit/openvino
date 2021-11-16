# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.opset8 as ops

from openvino import Function
from openvino.impl.op import Parameter


def test_function_add_outputs_tensor_name():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
