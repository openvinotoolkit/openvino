# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import unittest

from openvino.tools.ovc.convert import convert_model


class TestLoggerState(unittest.TestCase):
    def test_failed_conversion_keeps_the_root_logger_level(self):
        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.INFO)
        try:
            with tempfile.NamedTemporaryFile(suffix=".pb") as model:
                model.write(b"not a model")
                model.flush()
                with self.assertRaises(Exception):
                    convert_model(model.name)

            self.assertEqual(root.level, logging.INFO)
        finally:
            root.setLevel(level)
