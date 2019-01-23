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

import onnx
from generator import generator, generate

from extensions.front.onnx.flatten_ext import FlattenFrontExtractor
from mo.ops.flatten_onnx import FlattenONNX
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


@generator
class TestFlattenONNXExt(unittest.TestCase):
    @staticmethod
    def _create_flatten_node(axis):
        pb = onnx.helper.make_node(
            'Flatten',
            inputs=['a'],
            outputs=['b'],
            axis=axis,
        )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Flatten'] = FlattenONNX

    @generate(*[x for x in range(4)])
    def test_flatten_ext(self, axis):
        node = self._create_flatten_node(axis)
        FlattenFrontExtractor.extract(node)

        exp_res = {
            'type': 'Reshape',
            'axis': axis,
            'infer': FlattenONNX.infer
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
