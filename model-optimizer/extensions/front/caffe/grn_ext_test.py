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

from extensions.front.caffe.grn_ext import GRNFrontExtractor
from extensions.ops.grn import GRNOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.ops.op import Op


class FakeGRNProtoLayer:
    def __init__(self, val):
        self.grn_param = val


class TestGRNExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['GRN'] = GRNOp

    def test_grn_no_pb_no_ml(self):
        self.assertRaises(AttributeError, GRNFrontExtractor.extract, None)

    @patch('extensions.front.caffe.grn_ext.merge_attrs')
    def test_grn_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'bias': 0.7
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeGRNProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        GRNFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "GRN",
            'bias': 0.7,
            'infer': copy_shape_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
