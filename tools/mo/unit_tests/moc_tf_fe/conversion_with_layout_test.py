# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
from generator import generator, generate

import openvino.runtime.opset11 as opset11
from openvino.runtime import Model
from openvino.runtime import PartialShape, Dimension
from openvino.tools.mo.convert import convert_model
from openvino.tools.mo.utils.error import Error


@generator
class TestConversionWithBatchAndLayout(unittest.TestCase):
    def basic_check(self, model_name: str, batch: int, layout: str, refs_shapes: dict):
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", model_name)
        ov_model = convert_model(input_model, batch=batch, layout=layout)

        for ov_input in ov_model.inputs:
            input_name = ov_input.any_name
            assert input_name in refs_shapes, "No reference input shape is found for {}".format(input_name)
            input_shape = ov_input.get_partial_shape()
            ref_shape = refs_shapes[input_name]
            assert input_shape == ref_shape, "Incorrect shape for {} input:" \
                                             " expected shape - {}, actual shape - {}".format(input_name, ref_shape,
                                                                                              input_shape)

    @unittest.skip("Fix importing of openvino.test_utils in Jenkins")
    def test_basic_model_no_layout(self):
        from openvino.test_utils import compare_functions
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", "model_fp32.pbtxt")
        ov_model = convert_model(input_model)

        # compare with the reference graph
        param1 = opset11.parameter([2, 2], name="in1", dtype=np.float32)
        param2 = opset11.parameter([2, 2], name="in2", dtype=np.float32)
        add = opset11.add(param1, param2, name="add")
        ref_model = Model(add, [param1, param2])
        flag, msg = compare_functions(ov_model, ref_model, compare_tensor_names=False)
        assert flag, msg

    @generate(
        *[
            (
                    "model_fp32.pbtxt", 5, "in1(cn),in2(cn)",
                    {"in1": PartialShape([2, 5]), "in2": PartialShape([2, 5])},
            ),
            (
                    "model_fp32.pbtxt", 9, "in1(nc),in2(nc)",
                    {"in1": PartialShape([9, 2]), "in2": PartialShape([9, 2])},
            ),
            (
                    "model_fp32.pbtxt", 7, "in1(?c),in2(?c)",
                    {"in1": PartialShape([2, 2]), "in2": PartialShape([2, 2])},
            ),
        ],
    )
    def test_basic_model_with_layout(self, model_name: str, batch: int, layout: str, refs_shapes: dict):
        self.basic_check(model_name, batch, layout, refs_shapes)

    @generate(
        *[
            (
                    "model_with_convolution_dynamic_rank.pbtxt", 7, "x(n???),kernel(????)",
                    {"x": PartialShape([7, Dimension.dynamic(), Dimension.dynamic(), 3]),
                     "kernel": PartialShape([2, 2, 3, 1])},
            ),
            (
                    "model_with_convolution_dynamic_rank.pbtxt", 3, "x(???n),kernel(??n?)",
                    {"x": PartialShape([Dimension.dynamic(), Dimension.dynamic(), Dimension.dynamic(), 3]),
                     "kernel": PartialShape([2, 2, 3, 1])},
            ),
        ],
    )
    def test_model_with_convolution_dynamic_rank(self, model_name: str, batch: int, layout: str, refs_shapes: dict):
        self.basic_check(model_name, batch, layout, refs_shapes)

    @generate(
        *[
            (
                    "model_fp32.pbtxt", 17, "",
                    {},
            ),
        ],
    )
    def test_model_expected_failure(self, model_name: str, batch: int, layout: str, refs_shapes: dict):
        # try to override batch size by default index (without specifying layout)
        with self.assertRaisesRegex(Error,
                                    "When you use -b \(--batch\) option, Model Optimizer applies its value to the first "
                                    "element of the shape if it is equal to -1, 0 or 1\."):
            self.basic_check(model_name, batch, layout, refs_shapes)
