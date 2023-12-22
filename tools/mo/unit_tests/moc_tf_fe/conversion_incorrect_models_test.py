# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
import os
from generator import generator, generate

from openvino.tools.mo.convert import convert_model

@generator
class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    @generate(
        *[
            # the default frontend
            (
                    False, False, None
            ),
            (
                    False, False, "tf"
            ),
            # new frontend
            (
                    True, False, None
            ),
            (
                    True, False, "tf"
            ),
        ],
    )
    def test_conversion_fake_pb_model(self, use_new_frontend, use_legacy_frontend, framework):
        with self.assertRaisesRegex(Exception,
                                    "Internal error or inconsistent input model: the frontend supports frozen formats"
                                    " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."):
            path = os.path.dirname(__file__)
            input_model = os.path.join(path, "test_models", "fake.pb")

            convert_model(input_model,
                          use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                          framework=framework)

    @generate(
        *[
            # the default frontend
            (
                    False, False, None,
                    r"Framework name can not be deduced from the given options"
            ),
            (
                    False, False, "tf",
                    r"Internal error or inconsistent input model: the frontend supports frozen formats"
                    " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."
            ),
            # new frontend
            (
                    True, False, None,
                    r"Option \-\-use_new_frontend is specified but the Model Optimizer is unable to find new frontend"
            ),
            (
                    True, False, "tf",
                    r"Internal error or inconsistent input model: the frontend supports frozen formats"
                    " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."
            ),
        ],
    )
    def test_conversion_empty_model(self, use_new_frontend, use_legacy_frontend, framework, exp_reg_exp):
        with tempfile.NamedTemporaryFile(mode='w') as tmp, self.assertRaisesRegex(Exception,
                                                                                  exp_reg_exp):
            tmp.write("")
            convert_model(tmp.name,
                          use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                          framework=framework)

    @generate(
        *[
            # the default frontend
            (
                    False, False, None,
                    r"Framework name can not be deduced from the given options"
            ),
            (
                    False, False, "tf",
                    r"Internal error or inconsistent input model: the frontend supports frozen formats"
                    " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."
            ),
            # new frontend
            (
                    True, False, None,
                    r"Option \-\-use_new_frontend is specified but the Model Optimizer is unable to find new frontend"
            ),
            (
                    True, False, "tf",
                    r"Internal error or inconsistent input model: the frontend supports frozen formats"
                    " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."
            ),
        ],
    )
    def test_conversion_fake_model_with_no_ext(self, use_new_frontend, use_legacy_frontend, framework, exp_reg_exp):
        with tempfile.NamedTemporaryFile(mode='w') as tmp, self.assertRaisesRegex(Exception,
                                                                                  exp_reg_exp):
            tmp.write("1212234\n12312")
            convert_model(tmp.name,
                          use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                          framework=framework)
