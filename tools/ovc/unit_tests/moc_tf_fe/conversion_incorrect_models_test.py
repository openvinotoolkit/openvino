# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

from generator import generator, generate

from openvino.tools.ovc.convert import convert_model


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