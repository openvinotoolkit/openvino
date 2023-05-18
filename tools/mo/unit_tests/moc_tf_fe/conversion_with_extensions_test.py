# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import openvino.runtime.opset11 as ops
from openvino.frontend import ConversionExtension
from unit_tests.moc_tf_fe.utils import basic_check


class TestConversionWithExtensions(unittest.TestCase):
    def conversion_add_v2(self, node):
        x = node.get_input(0)
        y = node.get_input(1)
        return ops.multiply(x, y).outputs()

    def test_basic_conversion_extension(self):
        # test the converted model to make sure that it performs element-wise summation of operands
        basic_check(input_model="model_fp32.pbtxt", argv_input=None,
                    input_data={"in1": np.array([[2.0, 4.0], [12.0, 8.0]], dtype=np.float32),
                                "in2": np.array([[1.0, -2.0], [-6.0, 1.0]], dtype=np.float32)},
                    expected_dtype=np.float32, expected_value=np.array([[3.0, 2.0], [6.0, 9.0]], dtype=np.float32),
                    only_conversion=False)

        # make sure that the model performs element-wise multiplication after the conversion extension
        basic_check(input_model="model_fp32.pbtxt", argv_input=None,
                    input_data={"in1": np.array([[2.0, 4.0], [12.0, 8.0]], dtype=np.float32),
                                "in2": np.array([[1.0, -2.0], [-6.0, 1.0]], dtype=np.float32)},
                    expected_dtype=np.float32, expected_value=np.array([[2.0, -8.0], [-72.0, 8.0]], dtype=np.float32),
                    only_conversion=False, extensions=ConversionExtension("AddV2",
                                                                          self.conversion_add_v2))
