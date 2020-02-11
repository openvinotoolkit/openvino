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

from extensions.front.caffe.power_file_ext import PowerFileFrontExtractor
from extensions.ops.power_file import PowerFileOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.ops.op import Op


class FakePowerFileProtoLayer:
    def __init__(self, val):
        self.power_file_param = val


class TestPowerFileExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PowerFile'] = PowerFileOp

    def test_power_file_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PowerFileFrontExtractor.extract, None)

    @patch('extensions.front.caffe.power_file_ext.collect_attributes')
    def test_mvn_ext_ideal_numbers(self, collect_attributes_mock):
        params = {
            'normalize_variance': 'True',
            'across_channels': 'False',
            'eps': 1e-9
        }
        collect_attributes_mock.return_value = {
            'shift_file': 'some_file_path'
        }

        fake_pl = FakePowerFileProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        PowerFileFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "PowerFile",
            'shift_file': 'some_file_path',
            'infer': copy_shape_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
