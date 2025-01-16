# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.caffe.pooling_ext import PoolingFrontExtractor
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.ops.pooling import Pooling
from unit_tests.utils.extractors import PB, FakeMultiParam


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
            'global_pooling': True,
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
            'exclude_pad': True,
            'infer': Pooling.infer,
            'global_pool': True,
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
            'global_pooling': False,
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
            'exclude_pad': False,
            'infer': Pooling.infer,
            'global_pool': False,
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
            'global_pooling': True
        }
        node = PB({'pb': FakeProtoLayer(FakeMultiParam(params))})
        self.assertRaises(ValueError, PoolingFrontExtractor.extract, node)
