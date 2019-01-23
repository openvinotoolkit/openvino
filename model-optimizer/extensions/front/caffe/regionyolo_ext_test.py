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

from extensions.front.caffe.regionyolo_ext import RegionYoloFrontExtractor
from extensions.ops.regionyolo import RegionYoloOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeRegionYoloProtoLayer:
    def __init__(self, val, val_f):
        self.region_yolo_param = val
        self.flatten_param = val_f


class TestReorgYoloExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['RegionYolo'] = RegionYoloOp

    def test_reogyolo_no_pb_no_ml(self):
        self.assertRaises(AttributeError, RegionYoloFrontExtractor.extract, None)

    @patch('extensions.front.caffe.regionyolo_ext.merge_attrs')
    def test_reogyolo_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'coords': 4,
            'classes': 20,
            'num': 5,
            'do_softmax': 1,
            'anchors': 5,
            'mask': 5,
        }
        params_flatten = {
            'axis': 1,
            'end_axis': -1
        }
        merge_attrs_mock.return_value = {
            **params,
            **params_flatten
        }

        fake_pl = FakeRegionYoloProtoLayer(FakeMultiParam(params), FakeMultiParam(params_flatten))
        fake_node = FakeNode(fake_pl, None)

        RegionYoloFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "RegionYolo",
            'coords': 4,
            'classes': 20,
            'num': 5,
            'axis': 1,
            'end_axis': -1,
            'do_softmax': 1,
            'anchors': 5,
            'mask': 5,
            'infer': RegionYoloOp.regionyolo_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
