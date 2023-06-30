# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
from generator import generator, generate

from openvino.runtime import Core
from openvino.tools.mo.convert import convert_model


@generator
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

    def test_conversion_failure_fallback_default(self):
        self.basic("ctc_model_based.pbtxt", None, None, None, None,
                   None, None, True, True, False, False)

    @generate(
        *[
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
        ],
    )
    def test_conversion_model_with_undefined_constant(self, model_name, argv_input, inputs, expected, dtype,
                                                      use_new_frontend, use_legacy_frontend):
        self.basic(model_name, argv_input, inputs, dtype, expected, only_conversion=False,
                   input_model_is_text=True, use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend)
