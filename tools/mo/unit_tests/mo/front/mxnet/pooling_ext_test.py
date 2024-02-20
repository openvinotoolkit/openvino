# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.pooling_ext import PoolingFrontExtractor
from unit_tests.utils.extractors import PB


class TestPoolingShapesParsing(unittest.TestCase):
    def test_conv_ext_ideal_numbers(self):
        params = {'attrs': {
            "kernel": "(3, 4)",
            "stride": "(3, 2)",
            "pad": "(7, 8)",
            "pool_type": "max"
        }}

        node = PB({'symbol_dict': params})
        PoolingFrontExtractor.extract(node)
        exp_res = {
            'op': 'Pooling',
            'pad': np.array([[0, 0], [0, 0], [7, 7], [8, 8]]),
            'pad_spatial_shape': np.array([[7, 7], [8, 8]]),
            'stride': np.array([1, 1, 3, 2]),
            'window': np.array([1, 1, 3, 4]),
            'pool_method': 'max',
            'exclude_pad': False,
        }

        for key in exp_res.keys():
            if key in ('pad', 'stride', 'window', 'pad_spatial_shape'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])
