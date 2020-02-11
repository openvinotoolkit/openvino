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

import numpy as np
import onnx
from generator import generator, generate

import extensions.front.onnx.activation_ext as extractors
from extensions.ops.activation_ops import Elu
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB
from mo.utils.unittest.graph import build_graph


@generator
class ActivationOpsONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node(op_name: str):
        pb = onnx.helper.make_node(op_name, ["X"], ["Y"])
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs(op_name: str):
        # reference output Node attributes
        return (
            dict(
                op=op_name,
            )
        )

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            self.assertTrue(status, 'Mismatch for field {}, observed: {}, expected: {}'.format(key, out[key], ref[key]))

    @staticmethod
    def _extract(op_name):
        node = __class__._create_node(op_name)
        getattr(extractors, op_name + 'Extractor').extract(node)
        return node.graph.node[node.id]

    @generate(*['Abs', 'Acos', 'Asin', 'Atan', 'Cos', 'Cosh', 'Erf', 'Exp', 'Floor', 'Log', 'Not', 'Sigmoid', 'Sin',
                'Sinh', 'Tan', 'Tanh'])
    def test_default(self, op_name):
        ref = self._base_attrs(op_name)
        if ref['op'] == 'Not':
            ref['op'] = 'LogicalNot'
        out = self._extract(op_name)
        self._match(out, ref)


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
        extractors.EluExtractor.extract(node)

        exp_res = {
            'type': 'Elu',
            'alpha': alpha,
            'infer': Elu.infer
        }

        for key in exp_res.keys():
            self.assertEqual(node[key], exp_res[key])
