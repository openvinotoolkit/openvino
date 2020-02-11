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

from extensions.front.mxnet.pooling_ext import PoolingFrontExtractor
from mo.utils.unittest.extractors import PB


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
            'exclude_pad': 'false',
        }

        for key in exp_res.keys():
            if key in ('pad', 'stride', 'window', 'pad_spatial_shape'):
                np.testing.assert_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])
