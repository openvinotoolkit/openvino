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

from extensions.front.caffe.accum_ext import AccumFrontExtractor
from extensions.ops.accum import AccumOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeAccumProtoLayer:
    def __init__(self, val):
        self.accum_param = val


class TestAccumExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Accum'] = AccumOp

    def test_accum_no_pb_no_ml(self):
        self.assertRaises(AttributeError, AccumFrontExtractor.extract, None)

    @patch('extensions.front.caffe.accum_ext.collect_attributes')
    def test_accum_ext(self, collect_attributes_mock):
        params = {
            'top_height': 200,
            'top_width': 300,
            'size_divisible_by': 3,
            'have_reference': 'False',
        }
        collect_attributes_mock.return_value = {
            **params,
            'have_reference': 0
        }

        fake_pl = FakeAccumProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        AccumFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Accum",
            'top_height': 200,
            'top_width': 300,
            'size_divisible_by': 3,
            'have_reference': 0,
            'infer': AccumOp.accum_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
