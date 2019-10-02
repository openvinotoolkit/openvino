"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.caffe.prelu_ext import PreluFrontExtractor
from extensions.ops.prelu import PreluOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakePReLUProtoLayer:
    def __init__(self, val):
        self.prelu_param = val


class TestPreluExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PReLU'] = PreluOp

    def test_prelu_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PreluFrontExtractor.extract, None)

    @patch('extensions.front.caffe.prelu_ext.merge_attrs')
    def test_reogyolo_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'channel_shared': False
        }

        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePReLUProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        PreluFrontExtractor.extract(fake_node)

        exp_res = {
            'type': 'PReLU',
            'op': 'PReLU',
            'channel_shared': 0,
            'infer': PreluOp.prelu_shape_infer,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
