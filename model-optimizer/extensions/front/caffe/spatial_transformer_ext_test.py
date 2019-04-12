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

from extensions.front.caffe.spatial_transformer_ext import SpatialTransformFrontExtractor
from extensions.ops.spatial_transformer import SpatialTransformOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.ops.op import Op


class FakeSpatialTransformProtoLayer:
    def __init__(self, val):
        self.st_param = val


class TestSpatialTransformExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['SpatialTransformer'] = SpatialTransformOp

    def test_st_no_pb_no_ml(self):
        self.assertRaises(AttributeError, SpatialTransformFrontExtractor.extract, None)

    @patch('extensions.front.caffe.spatial_transformer_ext.merge_attrs')
    def test_st_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'transform_type': "ffff",
            'sampler_type': "gggg",
            'output_H': 56,
            'output_W': 78,
            'to_compute_dU': True,
            'theta_1_1': 0.1,
            'theta_1_2': 0.2,
            'theta_1_3': 0.3,
            'theta_2_1': 0.4,
            'theta_2_2': 0.5,
            'theta_2_3': 0.6
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeSpatialTransformProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        SpatialTransformFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "SpatialTransformer",
            'transform_type': "ffff",
            'sampler_type': "gggg",
            'output_H': 56,
            'output_W': 78,
            'to_compute_dU': 1,
            'theta_1_1': 0.1,
            'theta_1_2': 0.2,
            'theta_1_3': 0.3,
            'theta_2_1': 0.4,
            'theta_2_2': 0.5,
            'theta_2_3': 0.6,
            'infer': SpatialTransformOp.sp_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
