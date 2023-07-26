# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest


from openvino.tools.mo.convert import convert_model

class TestMoFreezePlaceholderTFFE(unittest.TestCase):
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

        for idx, (use_new_frontend, use_legacy_frontend, framework, exp_reg_exp) in enumerate(test_cases):
            with self.subTest(test_case=idx):
                with tempfile.NamedTemporaryFile(mode='w') as tmp:
                    tmp.write("")  # Write an empty string to simulate an empty model file
                    tmp.flush()
                    with self.assertRaisesRegex(Exception, exp_reg_exp):
                        convert_model(
                            tmp.name,
                            use_new_frontend=use_new_frontend,
                            use_legacy_frontend=use_legacy_frontend,
                            framework=framework
                        )

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

        for idx, (use_new_frontend, use_legacy_frontend, framework, exp_reg_exp) in enumerate(test_cases):
            with self.subTest(test_case=idx):
                with tempfile.NamedTemporaryFile(mode='w') as tmp:
                    tmp.write("1212234\n12312")
                    tmp.flush()
                    with self.assertRaisesRegex(Exception, exp_reg_exp):
                        convert_model(
                            tmp.name,
                            use_new_frontend=use_new_frontend,
                            use_legacy_frontend=use_legacy_frontend,
                            framework=framework
                        )
