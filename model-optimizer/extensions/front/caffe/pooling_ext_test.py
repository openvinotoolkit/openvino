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

from extensions.front.caffe.pooling_ext import PoolingFrontExtractor
from mo.front.common.extractors.utils import layout_attrs
from mo.ops.pooling import Pooling
from mo.utils.unittest.extractors import PB, FakeMultiParam


class FakeProtoLayer:
    def __init__(self, val):
        self.pooling_param = val


class TestPooling(unittest.TestCase):
    def test_pooling_ext_global(self):
        params = {
            'kernel_size': 1,
            'stride': 2,
            'pad': 3,
            'pool': 0,
            'global_pooling': 1,
            'ceil_mode': 1
        }
        node = PB({'pb': FakeProtoLayer(FakeMultiParam(params))})
        PoolingFrontExtractor.extract(node)
        res = node
        exp_res = {
            'window': np.array([1, 1, 0, 0], dtype=np.int64),
            'stride': np.array([1, 1, 1, 1], dtype=np.int64),
            'pad': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
            'pad_spatial_shape': np.array([[0, 0], [0, 0]], dtype=np.int64),
            'pool_method': 'max',
            'exclude_pad': 'true',
            'infer': Pooling.infer,
            'global_pool': 1,
            'output_spatial_shape': None,
            'pooling_convention': 'full',
            'rounding_type': 'ceil'

        }
        exp_res.update(layout_attrs())
        for i in exp_res.keys():
            if i in ('window', 'stride',
                     'pad', 'pad_spatial_shape',
                     'spatial_dims', 'batch_dims',
                     'channel_dims'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_pooling_ext(self):
        params = {
            'kernel_size': 1,
            'stride': 2,
            'pad': 3,
            'pool': 1,
            'global_pooling': 0,
            'ceil_mode': 0
        }
        node = PB({'pb': FakeProtoLayer(FakeMultiParam(params))})
        PoolingFrontExtractor.extract(node)
        res = node
        exp_res = {
            'window': np.array([1, 1, 1, 1], dtype=np.int64),
            'stride': np.array([1, 1, 2, 2], dtype=np.int64),
            'pad': np.array([[0, 0], [0, 0], [3, 3], [3, 3]], dtype=np.int64),
            'pad_spatial_shape': np.array([[3, 3], [3, 3]], dtype=np.int64),
            'pool_method': 'avg',
            'exclude_pad': 'false',
            'infer': Pooling.infer,
            'global_pool': 0,
            'output_spatial_shape': None,
            'pooling_convention': 'valid'
        }
        exp_res.update(layout_attrs())
        for i in exp_res.keys():
            if i in ('window', 'stride',
                     'pad', 'pad_spatial_shape',
                     'spatial_dims', 'batch_dims',
                     'channel_dims'):
                np.testing.assert_array_equal(res[i], exp_res[i])
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_pooling_ext_exception(self):
        params = {
            'kernel_size': 1,
            'stride': 2,
            'pad': 3,
            'pool': 3,
            'global_pooling': 1
        }
        node = PB({'pb': FakeProtoLayer(FakeMultiParam(params))})
        self.assertRaises(ValueError, PoolingFrontExtractor.extract, node)
