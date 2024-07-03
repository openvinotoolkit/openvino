# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from openvino.tools.mo.convert import convert_model


class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    def test_conversion_fake_pb_model(self):
        test_cases = [
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
        ]
        for use_new_frontend, use_legacy_frontend, framework in test_cases:
            with self.assertRaisesRegex(Exception,
                                        "Internal error or inconsistent input model: the frontend supports frozen formats"
                                        " \(.pb and .pbtxt\), SavedModel and MetaGraph \(.meta\), and v1 checkpoints."):
                path = os.path.dirname(__file__)
                input_model = os.path.join(path, "test_models", "fake.pb")

                convert_model(input_model,
                              use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                              framework=framework)

    def test_conversion_empty_model(self):
        test_cases = [
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
        ]
        for use_new_frontend, use_legacy_frontend, framework, exp_reg_exp in test_cases:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False
            ) as tmp, self.assertRaisesRegex(Exception, exp_reg_exp):
                tmp.write("")
                # on Windows tmp file must be not deleted on close to avoid remove it when reopened by MO
                convert_model(tmp.name,
                              use_new_frontend=use_new_frontend, use_legacy_frontend=use_legacy_frontend,
                              framework=framework)
            os.remove(tmp.name)

    def test_conversion_fake_model_with_no_ext(self):
        test_cases = [
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
        ]
        for use_new_frontend, use_legacy_frontend, framework, exp_reg_exp in test_cases:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False
            ) as tmp, self.assertRaisesRegex(Exception, exp_reg_exp):
                tmp.write("1212234\n12312")
                # on Windows tmp file must be not deleted on close to avoid remove it when reopened by MO
                convert_model(
                    tmp.name,
                    use_new_frontend=use_new_frontend,
                    use_legacy_frontend=use_legacy_frontend,
                    framework=framework,
                )
            os.remove(tmp.name)
