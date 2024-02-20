# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value
from openvino.tools.mo.utils.utils import match_shapes


class TestMatchShapes(unittest.TestCase):
    def run_match_shapes(self, pattern: list, shape: list):
        return match_shapes(shape_array(pattern), shape_array(shape))

    def test_positive(self):
        self.assertTrue(self.run_match_shapes([], []))
        self.assertTrue(self.run_match_shapes([1, 2, 3], [1, 2, 3]))
        self.assertTrue(self.run_match_shapes([dynamic_dimension_value, 2, 3], [1, 2, 3]))
        self.assertTrue(self.run_match_shapes([1, dynamic_dimension_value, 3], [1, 2, 3]))
        self.assertTrue(self.run_match_shapes([dynamic_dimension_value, dynamic_dimension_value,
                                               dynamic_dimension_value], [1, 2, 3]))
        self.assertTrue(self.run_match_shapes([dynamic_dimension_value], [2]))

    def test_negative(self):
        self.assertFalse(self.run_match_shapes([dynamic_dimension_value], []))
        self.assertFalse(self.run_match_shapes([dynamic_dimension_value], [1, 2, 3]))
        self.assertFalse(self.run_match_shapes([dynamic_dimension_value, 2, 3], [1, 3, 3]))
        self.assertFalse(self.run_match_shapes([1, dynamic_dimension_value, 3], [2, 2]))
        self.assertFalse(self.run_match_shapes([dynamic_dimension_value, dynamic_dimension_value,
                                                dynamic_dimension_value], [2, 3, 4, 5]))
