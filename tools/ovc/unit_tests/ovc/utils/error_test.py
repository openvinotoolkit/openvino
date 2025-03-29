# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.ovc.error import classify_error_type


class TestingErrorClassifier(unittest.TestCase):
    def test_no_module(self):
        message = "No module named 'openvino._offline_transformations.offline_transformations_api'"
        self.assertEqual(classify_error_type(message), message)

    def test_no_module_neg(self):
        message = "No module 'openvino'"
        self.assertEqual(classify_error_type(message), "undefined")

    def test_cannot_import_name(self):
        message = "cannot import name 'IECore' from 'openvino.inference_engine' (unknown location)"
        self.assertEqual(classify_error_type(message), "cannot import name 'IECore'")

    def test_cannot_import_name_neg(self):
        message = "import name 'IECore' from 'openvino.inference_engine' (unknown location)"
        self.assertEqual(classify_error_type(message), "undefined")
