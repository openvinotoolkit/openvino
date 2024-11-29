# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import unittest
from openvino.runtime import Core
from openvino.tools.mo.convert import convert_model
from sys import platform

# TODO: Segfault on CPU CVS-154874
@unittest.skip("Segfault on CPU CVS-154874")
class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    def basic(self, input_model, argv_input, inputs, dtype, expected, freeze_placeholder_with_value=None,
              input_shape=None, only_conversion=False, input_model_is_text=True, use_new_frontend=True,
              use_legacy_frontend=False):
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", input_model)

        try:
            model = convert_model(input_model, input=argv_input,
                                  freeze_placeholder_with_value=freeze_placeholder_with_value,
                                  input_shape=input_shape, input_model_is_text=input_model_is_text,
                                  use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                                  framework="tf")
        except Exception as ex:
            self.fail("Model conversion failed due to error: {}".format(ex))

        if only_conversion:
            return

        ie = Core()
        exec_net = ie.compile_model(model, "CPU")
        req = exec_net.create_infer_request()
        results = req.infer(inputs)
        values = list(results.values())[0]
        if dtype is not None:
            assert values.dtype == dtype
        assert np.allclose(values, expected)

    def test_fp32(self):
        test_cases = [
            (
                "in1[1 4]->[1.0 2.0 3.0 4.0],in2[1 4]{f32}->[1.0 2.0 3.0 4.0]",
                {},
                np.array([2.0, 4.0, 6.0, 8.0]),
                np.float32,
            ),
            (
                "in2{f32}->[0.0 0.0 0.0 0.0]",
                {"in1:0": np.array([[1.0, 2.0], [3.0, 4.0]])},
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.float32,
            ),
            (
                "in2->[1.0 15.0 15.5 1.0]",
                {"in1:0": np.array([[2.0, 4.0], [12.0, 8.0]])},
                np.array([[3.0, 19.0], [27.5, 9.0]]),
                np.float32,
            ),
            (
                "in1[1 4]{i32}->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                {},
                np.array([2.0, 4.0, 6.0, 8.0]),
                np.int32,
            ),
        ]
        for input_freezing_value, inputs, expected, dtype in test_cases:
            self.basic("model_fp32.pbtxt", input_freezing_value, inputs, dtype, expected)

    def test_int32(self):
        test_cases = [
            (
                "in1[1 4]->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                {},
                np.array([1, 4, 9, 16]),
                np.int32,
            ),
            (
                "in2->[2 5 6 7 3 2]",
                {"in1:0": np.array([[2, 4, 1], [1, 2, 8]])},
                np.array([[4, 20, 6], [7, 6, 16]]),
                np.int32,
            ),
        ]
        for input_freezing_value, inputs, expected, dtype in test_cases:
            self.basic("model_int32.pbtxt", input_freezing_value, inputs, dtype, expected)

    def test_bool(self):
        test_cases = [
            (
                "in1[2]->[True False],in2[2]->[True True]",
                {},
                np.array([True, False], dtype=bool),
                bool,
            ),
            (
                "in2[2,3]->[True,True,False,True,True,False]",
                {"in1:0": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                np.array([[False, True, False], [False, True, False]], dtype=bool),
                bool,
            ),
            (
                "in2[]->True",
                {"in1:0": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                np.array([[False, True, True], [False, True, True]], dtype=bool),
                bool,
            ),
        ]
        for input_freezing_value, inputs, expected, dtype in test_cases:
            self.basic("model_bool.pbtxt", input_freezing_value, inputs, dtype, expected)

    def test_bool2(self):
        test_cases = [
            (
                "in1[3]->[1 2 3],in2[3]->[4 5 6],cond->False",
                {},
                np.array([4, 5, 6], dtype=np.float32),
                np.float32,
                None,
                None,
                False
            ),
            (
                None,
                {"in1:0": np.array([2.0, 4.0, 6.0], dtype=np.float32),
                 "in2:0": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                np.array([2, 4, 6], dtype=np.float32),
                np.float32,
                "cond:0->False",
                None,
                True  # fill a bug to investigate why compilation of this model is hang on
            ),
            # case: input_shape + freeze_placeholder_with_value
            (
                None,
                {"in2:0": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                np.array([2, 4, 6], dtype=np.float32),
                np.float32,
                "in1:0->[2.0 4.0 6.0],cond:0->True",
                "[3]",
                False
            ),
        ]
        for input_freezing_value, inputs, expected, dtype, freeze_placeholder_with_value, \
            input_shape, only_conversion in test_cases:
            self.basic("model_bool2.pbtxt", input_freezing_value, inputs, dtype, expected,
                       freeze_placeholder_with_value,
                       input_shape, only_conversion)

    def test_cutting_fp32(self):
        test_cases = [
            (
                "add:0[3],z:0",
                {"add:0": np.array([4, 5, 6], dtype=np.float32), "z:0": np.array([1, 2, 3], dtype=np.float32)},
                np.array([4, 10, 18], dtype=np.float32),
                np.float32,
                None,
                None,
                False
            ),
            (
                "add:0{i32}[3],z:0{i32}",
                {"add:0": np.array([4, 5, 6], dtype=np.int32), "z:0": np.array([1, 2, 3], dtype=np.int32)},
                np.array([4, 10, 18], dtype=np.int32),
                np.int32,
                None,
                None,
                False
            ),
        ]
        for input_freezing_value, inputs, expected, dtype, freeze_placeholder_with_value, \
            input_shape, only_conversion in test_cases:
            self.basic("model_three_inputs.pbtxt", input_freezing_value, inputs, dtype, expected,
                       freeze_placeholder_with_value,
                       input_shape, only_conversion, True)

    def test_placeholder_with_default(self):
        test_cases = [
            (
                "x[1,4],y[4]",
                {"x": np.array([[3, 2, 1, 5]], dtype=np.int32), "y": np.array([0, -1, -7, 8], dtype=np.int32)},
                np.array([[3, 1, -6, 13]], dtype=np.int32),
                np.int32,
                None,
                None,
                False
            ),
            (
                "x,y",
                {"x": np.array([[-3, 20, 1]], dtype=np.int32), "y": np.array([[10, -11, -17]], dtype=np.int32)},
                np.array([[7, 9, -16]], dtype=np.int32),
                np.int32,
                None,
                None,
                False
            ),
            (
                "x",
                {"x": np.array([[-3, 20, 1]], dtype=np.int32)},
                np.array([[-2, 22, 4], [1, 25, 7]], dtype=np.int32),
                np.int32,
                None,
                None,
                False
            ),
        ]
        for inputs, inputs_data, expected, dtype, freeze_placeholder_with_value, \
            input_shape, only_conversion in test_cases:
            self.basic("placeholder_with_default.pbtxt", inputs, inputs_data, dtype, expected,
                       freeze_placeholder_with_value,
                       input_shape, only_conversion, True)

    def test_freeze_placeholder_with_unknown_rank(self):
        test_cases = [
            (
                "x[4],y->2.0",
                {"x": np.array([3, 2, 1, 5], dtype=np.float32)},
                np.array([6, 4, 2, 10], dtype=np.float32),
                np.float32,
                None,
                None,
                False
            ),
            (
                "x[1],y->[2.0,3.0]",
                {"x": np.array([3], dtype=np.float32)},
                np.array([6, 9], dtype=np.float32),
                np.float32,
                None,
                None,
                False
            ),
        ]
        for inputs, inputs_data, expected, dtype, freeze_placeholder_with_value, \
            input_shape, only_conversion in test_cases:
            self.basic("mul_with_unknown_rank_y.pbtxt", inputs, inputs_data, dtype, expected,
                       freeze_placeholder_with_value,
                       input_shape, only_conversion, True)

    def test_conversion_tf1_while_default(self):
        self.basic("ctc_model_based.pbtxt", None, None, None, None,
                   None, None, True, True, False, False)

    def test_conversion_tf1_while_use_new_frontend(self):
        self.basic("ctc_model_based.pbtxt", None, None, None, None,
                   None, None, True, True, True, False)

    @unittest.skip("88349: Fix auto-pruning in legacy FE")
    def test_conversion_model_oneshot_iterator_use_legacy_frontend(self):
        self.basic("model_oneshot_iterator.pbtxt", None, None, None, None,
                   None, None, True, True, False, True)

    def test_conversion_model_oneshot_iterator_default(self):
        self.basic("model_oneshot_iterator.pbtxt", None, None, None, None,
                   None, None, True, True, False, False)

    @unittest.skip("109220: Use generating script for this test model instead of Git LFS")
    def test_conversion_model_with_non_standard_extension(self):
        test_cases = [
            (
                "in2{f32}->[0.0 0.0 0.0 0.0]",
                {"in1": np.array([[1.0, 2.0], [3.0, 4.0]])},
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.float32,
            ),
            (
                "in2->[1.0 15.0 15.5 1.0]",
                {"in1": np.array([[2.0, 4.0], [12.0, 8.0]])},
                np.array([[3.0, 19.0], [27.5, 9.0]]),
                np.float32,
            ),
        ]
        for input_freezing_value, inputs, expected, dtype in test_cases:
            self.basic("model_fp32.frozen", input_freezing_value, inputs, dtype, expected, only_conversion=False,
                       input_model_is_text=False, use_new_frontend=True,
                       use_legacy_frontend=False)

    @unittest.skip("109220: Make TF FE to return the error")
    def test_conversion_dir_model(self):
        with self.assertRaisesRegex(Exception,
                                    "Internal error or inconsistent input model: the frontend supports "
                                    "only frozen binary protobuf format."):
            self.basic(".", None, None, None, None,
                       only_conversion=True, input_model_is_text=False, use_new_frontend=True,
                       use_legacy_frontend=False)

    def test_conversion_pbtxt_model_with_inference(self):
        test_cases = [
            (
                {"x:0": np.array([1, 2], dtype=np.int32), "y:0": np.array([4], dtype=np.int32)},
                np.array([-3, -2], dtype=np.int32),
                np.int32,
            ),
            (
                {"x:0": np.array([20, 25], dtype=np.int32), "y:0": np.array([10], dtype=np.int32)},
                np.array([30, 35], dtype=np.int32),
                np.int32,
            )
        ]
        for inputs, expected, dtype in test_cases:
            self.basic("model_with_if.pbtxt", None, inputs, dtype, expected, only_conversion=False,
                       input_model_is_text=False, use_new_frontend=True, use_legacy_frontend=False)

    def test_conversion_model_with_undefined_constant(self):
        test_cases = [
            # legacy frontend
            (
                "model_add_with_undefined_constant.pbtxt",
                "x[2,3]",
                {"x": np.array([[2, 3, 0], [1, 4, 6]], dtype=np.float32)},
                np.array([[2, 3, 0], [1, 4, 6]], dtype=np.float32),
                np.float32, False, True,
            ),
            (
                "model_mul_with_undefined_constant.pbtxt",
                "x[2]",
                {"x": np.array([-1, 2], dtype=np.int32)},
                np.array([0, 0], dtype=np.int32),
                np.int32, False, True,
            ),
            # new frontend
            (
                "model_add_with_undefined_constant.pbtxt",
                "x[2,3]",
                {"x": np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32)},
                np.array([[12, 13, 10], [11, 14, 16]], dtype=np.float32),
                np.float32, True, False,
            ),
            (
                "model_mul_with_undefined_constant.pbtxt",
                "x[2]",
                {"x": np.array([11, -12], dtype=np.int32)},
                np.array([0, 0], dtype=np.int32),
                np.int32, True, False,
            ),
        ]
        for model_name, argv_input, inputs, expected, dtype, use_new_frontend, use_legacy_frontend in test_cases:
            self.basic(model_name, argv_input, inputs, dtype, expected, only_conversion=False,
                       input_model_is_text=True, use_new_frontend=use_new_frontend,
                       use_legacy_frontend=use_legacy_frontend)
