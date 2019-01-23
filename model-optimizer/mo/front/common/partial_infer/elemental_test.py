"""
 Copyright (c) 2018 Intel Corporation

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
from unittest.mock import patch

import numpy as np

from mo.front.common.partial_infer.elemental import copy_shape_infer


class FakeNode:
    def __init__(self, blob):
        self.blob = blob

    def in_shape(self):
        return self.blob


class TestElementalInference(unittest.TestCase):
    @patch('mo.front.common.partial_infer.elemental.single_output_infer')
    def test_copy_shape_infer(self, single_output_infer_mock):
        single_output_infer_mock.return_value = 0
        node = FakeNode(np.array([1, 2]))
        copy_shape_infer(node)
        self.assertTrue(single_output_infer_mock.called)
