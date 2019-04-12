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

from extensions.front.caffe.interp_ext import InterpFrontExtractor
from extensions.ops.interp import InterpOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeInterpProtoLayer:
    def __init__(self, val):
        self.interp_param = val


class TestInterpExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Interp'] = InterpOp

    def test_interp_no_pb_no_ml(self):
        self.assertRaises(AttributeError, InterpFrontExtractor.extract, None)

    @patch('extensions.front.caffe.interp_ext.merge_attrs')
    def test_interp_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'height': 1.1,
            'width': 2.2,
            'zoom_factor': 3.3,
            'shrink_factor': 4.4,
            'pad_beg': 5.5,
            'pad_end': 6.6
        }
        merge_attrs_mock.return_value = {
            **params,
            'test': 54,
            'test2': 'test3'
        }

        fake_pl = FakeInterpProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)
        InterpFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Interp",
            'height': 1.1,
            'width': 2.2,
            'zoom_factor': 3.3,
            'shrink_factor': 4.4,
            'pad_beg': 5.5,
            'pad_end': 6.6,
            'infer': InterpOp.interp_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
