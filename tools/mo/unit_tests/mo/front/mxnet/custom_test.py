# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.custom import CustomFrontExtractorOp
from openvino.tools.mo.front.extractor import FrontExtractorOp, MXNetCustomFrontExtractorOp
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

attrs = {'test_attr': 1}


class FakeExtractor(MXNetCustomFrontExtractorOp):
    @classmethod
    def extract(cls, node: Node):
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
        self.assertEqual(op_attrs, attrs)
