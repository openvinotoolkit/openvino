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

from extensions.front.mxnet.custom import CustomFrontExtractorOp
from mo.utils.unittest.graph import build_graph
from mo.front.extractor import FrontExtractorOp, MXNetCustomFrontExtractorOp
from mo.graph.graph import Node

attrs = {'test_attr': 1}


class FakeExtractor(MXNetCustomFrontExtractorOp):
    @staticmethod
    def extract(node: Node):
        return True, attrs


class TestCustomFrontExtractorOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FrontExtractorOp.registered_ops['Custom'] = CustomFrontExtractorOp

    def test_extract_custom_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
             'node_custom': {'type': 'Custom', 'value': None, 'kind': 'op', 'op': 'Custom', },
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [('node_1', 'node_2'),
             ('node_2', 'node_custom'),
             ('node_custom', 'node_3'),
             ],
            {
                'node_custom': {'symbol_dict': {'attrs': {'op_type': 'test_type'}}},
            })

        custom_node = Node(graph, 'node_custom')
        custom_op = FakeExtractor()
        supported, op_attrs = custom_op.extract(custom_node)
        self.assertTrue(supported)
        self.assertEquals(op_attrs, attrs)
