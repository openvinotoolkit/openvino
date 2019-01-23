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

from mo.front.caffe.extractors.power import power_ext
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.unittest.extractors import FakeMultiParam


class FakeProtoLayer:
    def __init__(self, val):
        self.power_param = val


class TestPowerExt(unittest.TestCase):
    def test_power_ext(self):
        params = {
            'power': 1,
            'scale': 2,
            'shift': 3
        }
        res = power_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'power': 1,
            'scale': 2,
            'shift': 3,
            'infer': copy_shape_infer,
            'op': "Power",
            'type': 'Power',
            'output_spatial_shape': None,
        }
        self.assertEqual(res, exp_res)
