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

import numpy as np
import onnx

from extensions.front.onnx.affine_ext import AffineFrontExtractor
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class AffineONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node(attrs: dict):
        pb = onnx.helper.make_node("Affine", ["X"], ["Y"], **attrs)
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs():
        # Commonly used attributes in the tests
        # Each test takes these ones and then adds/modifies/deletes particular fields
        return (
            # test input ONNX attributes
            dict(
                alpha=1.0,
                beta=0.0
            ),
            # reference output Node attributes
            dict(
                op='ImageScaler',
                scale=1.0,
                bias=0.0
            )
        )

    @staticmethod
    def _extract(inp):
        node = __class__._create_node(inp)
        AffineFrontExtractor.extract(node)
        return node.graph.node[node.id]

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            self.assertTrue(status, 'Mismatch for field {}, observed: {}, expected: {}'.format(key, out[key], ref[key]))

    def test_default(self):
        inp, ref = self._base_attrs()
        out = self._extract(inp)
        self._match(out, ref)

    def test_random(self):
        inp, ref = self._base_attrs()
        inp['alpha'] = 123.
        inp['beta'] = 321.

        ref['scale'] = 123.
        ref['bias'] = 321.

        out = self._extract(inp)
        self._match(out, ref)
