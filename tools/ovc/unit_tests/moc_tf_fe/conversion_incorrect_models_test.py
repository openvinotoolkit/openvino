# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

from openvino.tools.ovc.convert import convert_model


class TestMoFreezePlaceholderTFFE(unittest.TestCase):
    def test_conversion_fake_pb_model(self):
        # TODO: Should FEs give detailed report why a model is rejected and should we print out the report?
        with self.assertRaisesRegex(Exception, "Cannot recognize input model."):
            path = os.path.dirname(__file__)
            input_model = os.path.join(path, "test_models", "fake.pb")
            convert_model(input_model)
