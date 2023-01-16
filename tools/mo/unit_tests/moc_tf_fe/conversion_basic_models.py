# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import unittest
from unittest.mock import Mock

import numpy as np
from generator import generator, generate

from openvino.frontend import (
    FrontEndManager,
    FrontEnd,
)  # pylint: disable=no-name-in-module,import-error
from openvino.runtime import Core
from openvino.tools.mo.convert_impl import prepare_ir


def base_args_config():
    args = argparse.Namespace()
    args.feManager = FrontEndManager()
    args.extensions = None
    # use new TF FE
    args.use_legacy_frontend = False
    args.use_new_frontend = True
    args.framework = "tf"
    args.model_name = None
    args.input_model = None
    args.input_model_is_text = False
    args.input_checkpoint = None
    args.saved_model_dir = None
    args.input_meta_graph = None
    args.saved_model_tags = None
    args.silent = True
    args.transform = []
    args.scale = None
    args.output = None
    args.input = None
    args.input_shape = None
    args.batch = None
    args.mean_values = None
    args.scale_values = None
    args.output_dir = os.getcwd()
    args.freeze_placeholder_with_value = None
    args.transformations_config = None
    args.static_shape = None
    args.reverse_input_channels = None
    args.data_type = None
    args.layout = None
    args.source_layout = None
    args.target_layout = None
    return args


try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


@generator
class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()
        FrontEnd.add_extension = Mock()

    def basic(self, input_model, argv_input, inputs, dtype, expected, freeze_placeholder_with_value=None,
              input_shape=None, only_conversion=False, input_model_is_text=True):
        path = os.path.dirname(__file__)
        input_model = os.path.join(path, "test_models", input_model)
        args = base_args_config()
        args.input_model = input_model
        args.input = argv_input
        args.freeze_placeholder_with_value = freeze_placeholder_with_value
        args.input_shape = input_shape
        args.input_model_is_text = input_model_is_text

        try:
            _, model = prepare_ir(args)
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

    @generate(
        *[
            (
                    "in1[1 4]->[1.0 2.0 3.0 4.0],in2[1 4]{f32}->[1.0 2.0 3.0 4.0]",
                    {},
                    np.array([2.0, 4.0, 6.0, 8.0]),
                    np.float32,
            ),
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
            (
                    "in1[1 4]{i32}->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                    {},
                    np.array([2.0, 4.0, 6.0, 8.0]),
                    np.int32,
            ),
        ],
    )
    def test_fp32(self, input_freezing_value, inputs, expected,
                  dtype):
        self.basic("model_fp32.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[1 4]->[1 2 3 4],in2[1 4]{i32}->[1 2 3 4]",
                    {},
                    np.array([1, 4, 9, 16]),
                    np.int32,
            ),
            (
                    "in2->[2 5 6 7 3 2]",
                    {"in1": np.array([[2, 4, 1], [1, 2, 8]])},
                    np.array([[4, 20, 6], [7, 6, 16]]),
                    np.int32,
            ),
        ],
    )
    def test_int32(self, input_freezing_value, inputs, expected,
                   dtype=None):
        self.basic("model_int32.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[2]->[True False],in2[2]->[True True]",
                    {},
                    np.array([True, False], dtype=bool),
                    bool,
            ),
            (
                    "in2[2,3]->[True,True,False,True,True,False]",
                    {"in1": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                    np.array([[False, True, False], [False, True, False]], dtype=bool),
                    bool,
            ),
            (
                    "in2[]->True",
                    {"in1": np.array([[False, True, True], [False, True, True]], dtype=bool)},
                    np.array([[False, True, True], [False, True, True]], dtype=bool),
                    bool,
            ),
        ],
    )
    def test_bool(self, input_freezing_value, inputs, expected,
                  dtype=None):
        self.basic("model_bool.pbtxt", input_freezing_value, inputs, dtype, expected)

    @generate(
        *[
            (
                    "in1[3]->[1 2 3],in2[3]->[4 5 6],cond->False",
                    {},
                    np.array([4, 5, 6], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    None,
                    {"in1": np.array([2.0, 4.0, 6.0], dtype=np.float32),
                     "in2": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                    np.array([2, 4, 6], dtype=np.float32),
                    np.float32,
                    "cond->False",
                    None,
                    True  # fill a bug to investigate why compilation of this model is hang on
            ),
            # case: input_shape + freeze_placeholder_with_value
            (
                    None,
                    {"in2": np.array([1.0, 3.0, 5.0], dtype=np.float32)},
                    np.array([2, 4, 6], dtype=np.float32),
                    np.float32,
                    "in1->[2.0 4.0 6.0],cond->True",
                    "[3]",
                    False
            ),
        ],
    )
    def test_bool2(self, input_freezing_value, inputs, expected,
                   dtype=None, freeze_placeholder_with_value=None, input_shape=None, only_conversion=False):
        self.basic("model_bool2.pbtxt", input_freezing_value, inputs, dtype, expected, freeze_placeholder_with_value,
                   input_shape, only_conversion)

    @generate(
        *[
            (
                    "add:0[3],z",
                    {"add:0": np.array([4, 5, 6], dtype=np.float32), "z": np.array([1, 2, 3], dtype=np.float32)},
                    np.array([4, 10, 18], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    "add:0{i32}[3],z{i32}",
                    {"add:0": np.array([4, 5, 6], dtype=np.int32), "z": np.array([1, 2, 3], dtype=np.int32)},
                    np.array([4, 10, 18], dtype=np.int32),
                    np.int32,
                    None
            ),
        ],
    )
    def test_cutting_fp32(self, input_freezing_value, inputs, expected,
                          dtype=None, freeze_placeholder_with_value=None, input_shape=None, only_conversion=False):
        self.basic("model_three_inputs.pbtxt", input_freezing_value, inputs, dtype, expected,
                   freeze_placeholder_with_value,
                   input_shape, only_conversion, True)

    @generate(
        *[
            (
                    "x[1,4],y[4]",
                    {"x": np.array([[3, 2, 1, 5]], dtype=np.int32), "y": np.array([0, -1, -7, 8], dtype=np.int32)},
                    np.array([[3, 1, -6, 13]], dtype=np.int32),
                    np.int32,
                    None
            ),
            (
                    "x,y",
                    {"x": np.array([[-3, 20, 1]], dtype=np.int32), "y": np.array([[10, -11, -17]], dtype=np.int32)},
                    np.array([[7, 9, -16]], dtype=np.int32),
                    np.int32,
                    None
            ),
            (
                    "x",
                    {"x": np.array([[-3, 20, 1]], dtype=np.int32)},
                    np.array([[-2, 22, 4], [1, 25, 7]], dtype=np.int32),
                    np.int32,
                    None
            ),
        ],
    )
    def test_placeholder_with_default(self, inputs, inputs_data, expected,
                                      dtype=None, freeze_placeholder_with_value=None, input_shape=None,
                                      only_conversion=False):
        self.basic("placeholder_with_default.pbtxt", inputs, inputs_data, dtype, expected,
                   freeze_placeholder_with_value,
                   input_shape, only_conversion, True)

    @generate(
        *[
            (
                    "x[4],y->2.0",
                    {"x": np.array([3, 2, 1, 5], dtype=np.float32)},
                    np.array([6, 4, 2, 10], dtype=np.float32),
                    np.float32,
                    None
            ),
            (
                    "x[1],y->[2.0,3.0]",
                    {"x": np.array([3], dtype=np.float32)},
                    np.array([6, 9], dtype=np.float32),
                    np.float32,
                    None
            ),
        ],
    )
    def test_freeze_placeholder_with_unknown_rank(self, inputs, inputs_data, expected,
                                                  dtype=None, freeze_placeholder_with_value=None, input_shape=None,
                                                  only_conversion=False):
        self.basic("mul_with_unknown_rank_y.pbtxt", inputs, inputs_data, dtype, expected,
                   freeze_placeholder_with_value,
                   input_shape, only_conversion, True)
