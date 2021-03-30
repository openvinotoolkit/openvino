# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
