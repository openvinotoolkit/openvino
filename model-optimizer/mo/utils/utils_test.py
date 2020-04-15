"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from mo.utils.utils import match_shapes


class TestMatchShapes(unittest.TestCase):

    def run_match_shapes(self, pattern: list, shape: list):
        return match_shapes(np.array(pattern, dtype=np.int64), np.array(shape, dtype=np.int64))

    def test_positive(self):
        self.assertTrue(self.run_match_shapes([], []))
        self.assertTrue(self.run_match_shapes([1,2,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1,2,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([1,-1,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1,-1,-1], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1], [2]))

    def test_negative(self):
        self.assertFalse(self.run_match_shapes([-1], []))
        self.assertFalse(self.run_match_shapes([-1], [1,2,3]))
        self.assertFalse(self.run_match_shapes([-1,2,3], [1,3,3]))
        self.assertFalse(self.run_match_shapes([1,-1,3], [2,2]))
        self.assertFalse(self.run_match_shapes([-1, -1, -1], [2, 3, 4, 5]))
