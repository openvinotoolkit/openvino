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

from mo.front.mxnet.extractors.eltwise import eltwise_ext
from mo.front.mxnet.extractors.utils import AttrDictionary


class TestEltwiseParsing(unittest.TestCase):
    def test_eltwise_sum(self):
        attrs = {}
        res = eltwise_ext(AttrDictionary(attrs), infer=lambda a, b: a + b, op_type="sum")
        exp_attrs = {
            'type': 'Eltwise',
            'operation': 'sum'
        }

        for key in exp_attrs.keys():
            self.assertEqual(res[key], exp_attrs[key])

    def test_eltwise_mul(self):
        attrs = {}
        res = eltwise_ext(AttrDictionary(attrs), infer=lambda a, b: a * b, op_type="mul")
        exp_attrs = {
            'type': 'Eltwise',
            'operation': 'mul'
        }

        for key in exp_attrs.keys():
            self.assertEqual(res[key], exp_attrs[key])
