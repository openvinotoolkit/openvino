# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.layout import get_batch_dim, get_width_dim, get_height_dim, get_features_dim, get_depth_dim, \
    shape_for_layout
from openvino.tools.mo.utils.error import Error


class TestLayoutFunctions(unittest.TestCase):
    def test_get_batch_dim_NCHW(self):
        self.assertEqual(get_batch_dim('NCHW', 4), 0)

    def test_get_batch_dim_NHWC(self):
        self.assertEqual(get_batch_dim('NHWC', 4), 0)

    def test_get_batch_dim_NCDHW(self):
        self.assertEqual(get_batch_dim('NCHW', 5), 0)

    def test_get_batch_dim_NDHWC(self):
        self.assertEqual(get_batch_dim('NHWC', 5), 0)

    def test_get_features_dim_NCHW(self):
        self.assertEqual(get_features_dim('NCHW', 4), 1)

    def test_get_features_dim_NHWC(self):
        self.assertEqual(get_features_dim('NHWC', 4), 3)

    def test_get_features_dim_NCDHW(self):
        self.assertEqual(get_features_dim('NCHW', 5), 1)

    def test_get_features_dim_NDHWC(self):
        self.assertEqual(get_features_dim('NHWC', 5), 4)

    def test_get_width_dim_NCHW(self):
        self.assertEqual(get_width_dim('NCHW', 4), 3)

    def test_get_width_dim_NHWC(self):
        self.assertEqual(get_width_dim('NHWC', 4), 2)

    def test_get_width_dim_NCDHW(self):
        self.assertEqual(get_width_dim('NCHW', 5), 4)

    def test_get_width_dim_NDHWC(self):
        self.assertEqual(get_width_dim('NHWC', 5), 3)

    def test_get_height_dim_NCHW(self):
        self.assertEqual(get_height_dim('NCHW', 4), 2)

    def test_get_height_dim_NHWC(self):
        self.assertEqual(get_height_dim('NHWC', 4), 1)

    def test_get_height_dim_NCDHW(self):
        self.assertEqual(get_height_dim('NCHW', 5), 3)

    def test_get_height_dim_NDHWC(self):
        self.assertEqual(get_height_dim('NHWC', 5), 2)

    def test_get_depth_dim_NCDHW(self):
        self.assertEqual(get_depth_dim('NCHW', 5), 2)

    def test_get_depth_dim_NDHWC(self):
        self.assertEqual(get_depth_dim('NHWC', 5), 1)

    def test_get_batch_dim_wrong_layout(self):
        self.assertRaises(AssertionError, get_batch_dim, 'NCDHW', 5)

    def test_get_width_dim_wrong_layout(self):
        self.assertRaises(AssertionError, get_width_dim, 'NCDHW', 5)

    def test_get_height_dim_wrong_layout(self):
        self.assertRaises(AssertionError, get_height_dim, 'NCDHW', 5)

    def test_get_features_dim_wrong_layout(self):
        self.assertRaises(AssertionError, get_features_dim, 'NCDHW', 5)

    def test_shape_for_layout_NCHW(self):
        self.assertListEqual([2, 3, 4, 5], list(shape_for_layout('NCHW', batch=2, features=3, height=4, width=5)))

    def test_shape_for_layout_NHWC(self):
        self.assertListEqual([2, 4, 5, 3], list(shape_for_layout('NHWC', batch=2, features=3, height=4, width=5)))

    def test_shape_for_layout_missing_batch(self):
        with self.assertRaises(Error):
            shape_for_layout('NCHW', features=3, height=4, width=5)

    def test_shape_for_layout_missing_features(self):
        with self.assertRaises(Error):
            shape_for_layout('NCHW', batch=2, height=4, width=5)

    def test_shape_for_layout_missing_height(self):
        with self.assertRaises(Error):
            shape_for_layout('NHWC', batch=2, features=3, width=5)

    def test_shape_for_layout_missing_width(self):
        with self.assertRaises(Error):
            shape_for_layout('NHWC', batch=2, features=3, height=4)

    def test_shape_for_layout_unknown_parameter(self):
        with self.assertRaises(Error):
            shape_for_layout('NHWC', batch=2, features=3, height=4, width=5, unknown_parameter=123)
