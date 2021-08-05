# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from mo.front.common.partial_infer.utils import int64_array, is_fully_defined, dynamic_dimension_value, \
    dynamic_dimension, shape_array, compare_shapes


def gen_masked_array(array, masked_indices):
    res = np.ma.masked_array(array)
    for index in masked_indices:
        res[index] = np.ma.masked
    return res


@generator
class IsFullyDefinedTest(unittest.TestCase):
    @generate(*[(None, False),
                (int64_array([2, 3, 5, 7]), True),  # int64 array with valid values
                (np.array([2, 3, 5, 7]), True),  # any numpy array with valid values
                (np.array([2, dynamic_dimension_value]), True),  # array with dynamic dimension value is fully defined!
                (gen_masked_array([2, 4, 5], [1]), False),  # masked array with at least one masked element
                (gen_masked_array([2, 4, 5], []), True),  # masked array with no masked elements is fully defined
                (dynamic_dimension, False),  # dynamic dimension is not fully defined
                (dynamic_dimension_value, True),  # dynamic dimension value is fully defined
                ((dynamic_dimension_value, dynamic_dimension_value), True),  # list with dynamic dimension values is
                # fully defined
                ((dynamic_dimension, 1), False),  # tuple with dynamic dimension is not fully defined
                ([dynamic_dimension, 1], False),  # list with dynamic dimension is not fully defined
                ])
    def test_is_fully_defined(self, data, result):
        self.assertEqual(is_fully_defined(data), result)


@generator
class ShapeArrayTest(unittest.TestCase):
    @generate(*[([1], gen_masked_array([1], []), True),
                # if we provide a list with dynamic_dimension_value then it is converted to dynamic dimension
                ([dynamic_dimension_value, 5], gen_masked_array([1, 5], [0]), True),
                # if we provide a list with dynamic_dimension then the generated shape array still have it
                ([7, dynamic_dimension], gen_masked_array([7, 1], [1]), True),
                # negative test to make sure that np.ma.allequal works properly
                ([2], gen_masked_array([1], []), False),
                ])
    def test_shape_array(self, data, ref, result):
        self.assertEqual(np.ma.allequal(shape_array(data), ref), result)


@generator
class CompareShapesTest(unittest.TestCase):
    @generate(*[(gen_masked_array([1, 2, 3], []), gen_masked_array([1, 2, 3], []), True),
                (gen_masked_array([4, 2, 3], []), gen_masked_array([1, 2, 3], []), False),
                (gen_masked_array([1, 2], []), gen_masked_array([1, 2, 3], []), False),
                (gen_masked_array([1, 2, 3], []), gen_masked_array([1, 2], []), False),
                (gen_masked_array([1, 2, 3], [1]), gen_masked_array([1, 5, 3], [1]), True),  # [1, d, 3] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], [2]), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, d] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], []), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, 3] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], [0]), gen_masked_array([1, 5, 3], []), False),  # [d, 2, 3] vs [1, 5, 3]
                (np.array([1, 2, 3]), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, 3] vs [1, d, 3]
                (np.array([1, 2]), gen_masked_array([1, 5, 3], [1]), False),
                (np.array([1, 2]), np.array([1, 2]), True),
                (np.array([1, 2]), np.array([3, 2]), False),
                ])
    def test_compare_shapes(self, input1, input2, result):
        self.assertEqual(compare_shapes(input1, input2), result)
