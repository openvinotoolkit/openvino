# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.OptimizeTransposeReshapeSequence import match_shapes, split_input_permute_dimension, \
    split_dims_indices, split_output_permute_dimension
from openvino.tools.mo.front.common.partial_infer.utils import int64_array


class SplitDimsIndicesTest(unittest.TestCase):
    def test_1(self):
        self.assertListEqual(list(split_dims_indices(int64_array([1, 32, 64, 60]), int64_array([1, 8, 4, 64, 3, 20]))), [1, 3])

    def test_2(self):
        self.assertListEqual(list(split_dims_indices(int64_array([8, 4, 64, 3, 20]), int64_array([1, 8, 4, 64, 3, 20, 1, 1]))), [0, 4, 4])

    def test_3(self):
        self.assertListEqual(list(split_dims_indices(int64_array([120]), int64_array([2, 3, 4, 1, 5]))), [0, 0, 0, 0])

    def test_4(self):
        self.assertListEqual(list(split_dims_indices(int64_array([120, 1]), int64_array([2, 3, 4, 5, 1]))), [0, 0, 0])

    def test_5(self):
        self.assertListEqual(list(split_dims_indices(int64_array([1, 4, 1, 1]), int64_array([1, 2, 1, 1, 2, 1, 1]))), [1, 1, 1])

    def test_6(self):
        self.assertListEqual(list(split_dims_indices(int64_array([1, 20, 64]), int64_array([1, 1, 20, 64]))), [1])


class SplitOutputTransposeDimensionTest(unittest.TestCase):
    def test_1(self):
        self.assertListEqual(list(split_output_permute_dimension(3, int64_array([0, 2, 3, 1]))), [0, 3, 4, 1, 2])

    def test_2(self):
        self.assertListEqual(list(split_output_permute_dimension(0, int64_array([0, 1, 3, 2]))), [0, 1, 2, 4, 3])

    def test_3(self):
        self.assertListEqual(list(split_output_permute_dimension(1, int64_array([0, 3, 1, 2]))), [0, 3, 4, 1, 2])


class SplitInputTransposeDimensionTest(unittest.TestCase):
    def test_1(self):
        self.assertListEqual(list(split_input_permute_dimension(1, int64_array([0, 2, 3, 1]))), [0, 3, 4, 1, 2])

    def test_2(self):
        self.assertListEqual(list(split_input_permute_dimension(0, int64_array([0, 1, 3, 2]))), [0, 1, 2, 4, 3])

    def test_3(self):
        self.assertListEqual(list(split_input_permute_dimension(3, int64_array([0, 3, 1, 2]))), [0, 3, 4, 1, 2])

    def test_4(self):
        self.assertListEqual(list(split_input_permute_dimension(0, int64_array([0, 1, 2, 3]))), [0, 1, 2, 3, 4])

    def test_5(self):
        self.assertListEqual(list(split_input_permute_dimension(3, int64_array([0, 1, 2, 3]))), [0, 1, 2, 3, 4])


class MatchShapesTest(unittest.TestCase):
    def test_basic(self):
        self.assertListEqual(list(match_shapes(int64_array([1, 32, 64, 60]), int64_array([8, 4, 64, 3, 20]))), [1, 8, 4, 64, 3, 20])

    def test_ones_in_the_middle(self):
        self.assertListEqual(list(match_shapes(int64_array([32, 1, 2, 3, 1, 8]), int64_array([4, 2, 1, 4, 6, 1, 1, 8]))), [4, 2, 1, 4, 1, 2, 3, 1, 1, 8])

    def test_trailing_one(self):
        self.assertListEqual(list(match_shapes(int64_array([1, 32, 64, 60, 1]), int64_array([8, 4, 64, 3, 20]))), [1, 8, 4, 64, 3, 20, 1])

    def test_one_to_many(self):
        self.assertListEqual(list(match_shapes(int64_array([120]), int64_array([2, 3, 4, 5]))), [2, 3, 4, 5])

    def test_many_to_one(self):
        self.assertListEqual(list(match_shapes(int64_array([2, 3, 4, 5]), int64_array([120]))), [2, 3, 4, 5])

    def test_many_to_one_with_trailing(self):
        self.assertListEqual(list(match_shapes(int64_array([2, 3, 4, 5]), int64_array([120, 1, 1]))), [2, 3, 4, 5, 1, 1])

    def test_equal_shapes(self):
        self.assertListEqual(list(match_shapes(int64_array([2, 3, 4, 5]), int64_array([2, 3, 4, 5]))), [2, 3, 4, 5])

    def test_one(self):
        self.assertListEqual(list(match_shapes(int64_array([1]), int64_array([1]))), [1])

    def test_ones_equal_lengths(self):
        self.assertListEqual(list(match_shapes(int64_array([1, 1, 1]), int64_array([1, 1, 1]))), [1, 1, 1])

    def test_ones_different_lengths(self):
        self.assertListEqual(list(match_shapes(int64_array([1]), int64_array([1, 1, 1]))), [1, 1, 1])

    def test_intersection_of_input_output_dimensions(self):  # is this test correct? Looks like yes...
        self.assertListEqual(list(match_shapes(int64_array([10, 20, 7]), int64_array([5, 4, 1, 70]))), [5, 2, 2, 1, 10, 7])

    def test_trailing_ones(self):
        self.assertListEqual(list(match_shapes(int64_array([1, 1, 10]), int64_array([1, 5, 1, 1, 2, 1]))), [1, 1, 5, 1, 1, 2, 1])

    def test_not_matchabale_shapes(self):
        self.assertIsNone(match_shapes(int64_array([5, 7]), int64_array([7, 5])))
