"""
 Copyright (C) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

from mo.utils.error import classify_error_type


class TestingErrorClassifier(unittest.TestCase):
    def test_no_module(self):
        message = "No module named 'openvino.offline_transformations.offline_transformations_api'"
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
