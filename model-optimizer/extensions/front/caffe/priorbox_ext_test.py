"""
 Copyright (c) 2018 Intel Corporation

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
from unittest.mock import patch

import numpy as np

from extensions.front.caffe.priorbox_ext import PriorBoxFrontExtractor
from extensions.ops.priorbox import PriorBoxOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakePriorBoxProtoLayer:
    def __init__(self, val):
        self.prior_box_param = val


class TestPriorBoxExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PriorBox'] = PriorBoxOp

    def test_priorbox_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PriorBoxFrontExtractor.extract, None)

    @patch('extensions.front.caffe.priorbox_ext.merge_attrs')
    def test_priorbox_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'clip': False,
            'flip': True,
            'min_size': np.array([]),
            'max_size': np.array([]),
            'aspect_ratio': np.array([2, 3]),
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePriorBoxProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        PriorBoxFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'PriorBox',
            'type': 'PriorBox',
            'clip': 0,
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }

        for key in exp_res.keys():
            if key in ['width', 'height', 'variance']:
                np.testing.assert_equal(fake_node[key], exp_res[key])
            else:
                self.assertEqual(fake_node[key], exp_res[key])
