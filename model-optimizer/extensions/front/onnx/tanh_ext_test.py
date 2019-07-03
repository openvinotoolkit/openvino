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

import numpy as np
import onnx

from extensions.front.onnx.tanh_ext import TanhFrontExtractor
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node


class TanhONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node():
        pb = onnx.helper.make_node("Tanh", ["X"], ["Y"])
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs():
        # Commonly used attributes in the tests
        # Each test takes these ones and then adds/modifies/deletes particular fields
        return (
            # reference output Node attributes
            dict(
                op='Activation',
                operation='tanh'
            )
        )

    @staticmethod
    def _extract():
        node = __class__._create_node()
        TanhFrontExtractor.extract(node)
        return node.graph.node[node.id]

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            self.assertTrue(status, 'Mismatch for field {}, observed: {}, expected: {}'.format(key, out[key], ref[key]))

    def test_default(self):
        ref = self._base_attrs()
        out = self._extract()
        self._match(out, ref)
