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

from extensions.front.caffe.normalize_ext import NormalizeFrontExtractor
from extensions.ops.normalize import NormalizeOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeNormalizeProtoLayer:
    def __init__(self, val):
        self.norm_param = val


class TestNormalizeExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Normalize'] = NormalizeOp

    def test_normalize_no_pb_no_ml(self):
        self.assertRaises(AttributeError, NormalizeFrontExtractor.extract, None)

    @patch('extensions.front.caffe.normalize_ext.collect_attributes')
    def test_normalize_ext_ideal_numbers(self, collect_attributes_mock):
        params = {
            'across_spatial': 1,
            'channel_shared': 0,
            'eps': 0.00001
        }
        collect_attributes_mock.return_value = {
            **params
        }

        fake_pl = FakeNormalizeProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        NormalizeFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Normalize",
            'across_spatial': 1,
            'channel_shared': 0,
            'eps': 0.00001,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
