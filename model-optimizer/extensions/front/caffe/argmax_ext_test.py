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
from unittest.mock import patch

from extensions.front.caffe.argmax_ext import ArgMaxFrontExtractor
from extensions.ops.argmax import ArgMaxOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeArgMaxProtoLayer:
    def __init__(self, val):
        self.argmax_param = val


class TestArgMaxExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['ArgMax'] = ArgMaxOp

    def test_argmax_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ArgMaxFrontExtractor.extract, None)

    @patch('extensions.front.caffe.argmax_ext.merge_attrs')
    def test_argmax_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'out_max_val': True,
            'top_k': 100,
            'axis': 2
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeArgMaxProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        ArgMaxFrontExtractor.extract(fake_node)

        exp_res = {
            'out_max_val': True,
            'top_k': 100,
            'axis': 2,
            'infer': ArgMaxOp.argmax_infer,
            'remove_values_output': True,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
