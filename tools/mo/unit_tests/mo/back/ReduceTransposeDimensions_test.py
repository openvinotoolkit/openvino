# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.ReduceTransposeDimensions import sequential_dims, merge_permute_order_dimensions, merge_dims
from openvino.tools.mo.front.common.partial_infer.utils import int64_array


class SequentialDimsTest(unittest.TestCase):
    def test_returns_first_instance(self):
        self.assertListEqual(sequential_dims(int64_array([0, 3, 4, 1, 2])), [1, 2])

    def test_returns_last_indices(self):
        self.assertListEqual(sequential_dims(int64_array([4, 0, 3, 1, 2])), [3, 4])

    def test_returns_full_list(self):
        self.assertListEqual(sequential_dims(int64_array([0, 1, 2, 3, 4])), [0, 1, 2, 3, 4])

    def test_returns_from_the_beginning(self):
        self.assertListEqual(sequential_dims(int64_array([1, 2, 3, 0, 4])), [0, 1, 2])

    def test_no_sequential_dims(self):
        self.assertIsNone(sequential_dims(int64_array([2, 1, 3, 0, 4])))

    def test_2d_input_with_sequential_dims(self):
        self.assertListEqual(sequential_dims(int64_array([0, 1])), [0, 1])

    def test_2d_input_without_sequential_dims(self):
        self.assertIsNone(sequential_dims(int64_array([1, 0])))


class MergeTransposeOrderDimensionsTest(unittest.TestCase):
    def test_merge_last_dims(self):
        self.assertListEqual(list(merge_permute_order_dimensions([1, 2], int64_array([0, 3, 4, 1, 2]))), [0, 3, 1, 2])

    def test_merge_last_indices(self):
        self.assertListEqual(list(merge_permute_order_dimensions([3, 4], int64_array([0, 3, 4, 1, 2]))), [0, 2, 3, 1])

    def test_merge_start_indices(self):
        self.assertListEqual(list(merge_permute_order_dimensions([0, 1], int64_array([1, 2, 4, 3, 0]))), [1, 3, 2, 0])

    def test_merge_all_dims(self):
        self.assertListEqual(list(merge_permute_order_dimensions([0, 1, 2], int64_array([0, 1, 2]))), [0])

    def test_merge_3_dims(self):
        self.assertListEqual(list(merge_permute_order_dimensions([1, 2, 3], int64_array([3, 0, 1, 2, 4]))), [1, 0, 2])


class MergeDimsTest(unittest.TestCase):
    def test_merge_middle_dims(self):
        self.assertListEqual(list(merge_dims([1, 2], int64_array([3, 2, 5, 7]))), [3, 10, 7])

    def test_merge_first_dim(self):
        self.assertListEqual(list(merge_dims([0, 1], int64_array([3, 2, 5, 7]))), [6, 5, 7])

    def test_merge_last_dim(self):
        self.assertListEqual(list(merge_dims([2, 3], int64_array([3, 2, 5, 7]))), [3, 2, 35])

    def test_merge_all_dims(self):
        self.assertListEqual(list(merge_dims([0, 1, 2, 3], int64_array([3, 2, 5, 7]))), [210])

    def test_reduce_with_minus_one(self):
        self.assertListEqual(list(merge_dims([1, 2], int64_array([3, -1, 5, 7]))), [3, -1, 7])

    def test_merge_with_0_being_merged(self):
        with self.assertRaisesRegex(AssertionError, ".*The value 0 is not supported.*"):
            merge_dims([1, 2], int64_array([3, 0, 5, 7]))

    def test_merge_with_0_not_merged(self):
        with self.assertRaisesRegex(AssertionError, ".*The value 0 is not supported.*"):
            merge_dims([2, 3], int64_array([3, 0, 5, 7]))
