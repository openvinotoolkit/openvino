"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.back.ReduceTransposeDimensions import sequential_dims, merge_permute_order_dimensions, merge_dims
from mo.front.common.partial_infer.utils import int64_array


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
