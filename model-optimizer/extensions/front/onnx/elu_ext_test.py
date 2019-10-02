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

import onnx
from generator import generator, generate

from extensions.front.onnx.elu_ext import EluFrontExtractor
from extensions.ops.activation_ops import Elu
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


@generator
class TestEluONNXExt(unittest.TestCase):
    @staticmethod
    def _create_elu_node(alpha=1.0):
        pb = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
            alpha=alpha
        )
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Elu'] = Elu

    @generate(*[1.0, 2.0, 3.0])
    def test_elu_ext(self, alpha):
        node = self._create_elu_node(alpha)
        EluFrontExtractor.extract(node)

        exp_res = {
            'type': 'Elu',
            'alpha': alpha,
            'infer': Elu.infer
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
