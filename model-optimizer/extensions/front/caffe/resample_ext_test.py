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

from extensions.front.caffe.resample_ext import ResampleFrontExtractor
from extensions.ops.resample import ResampleOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeResampleProtoLayer:
    def __init__(self, val):
        self.resample_param = val


class TestResampleExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Resample'] = ResampleOp

    def test_da_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ResampleFrontExtractor.extract, None)

    @patch('extensions.front.caffe.resample_ext.merge_attrs')
    def test_resample_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'antialias': True,
            'height': 384,
            'width': 512,
            'type': 2,
            'factor': 1.0,
        }
        merge_attrs_mock.return_value = {
            'antialias': True,
            'height': 384,
            'width': 512,
            'type': 'caffe.ResampleParameter.LINEAR',
            'factor': 1.0,
        }
        fake_pl = FakeResampleProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        ResampleFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'Resample',
            'antialias': 1,
            'height': 384,
            'width': 512,
            'resample_type': 'caffe.ResampleParameter.LINEAR',
            'factor': 1.0,
            'infer': ResampleOp.resample_infer
        }

        for key in exp_res.keys():
            self.assertEqual(exp_res[key], fake_node[key])
