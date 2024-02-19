# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.caffe.axpy import AxpyToSSandAdd
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph_with_edge_attrs


class TestAxpyReplacer(unittest.TestCase):
    def test_axpy(self):
        nodes = {
            'node_1': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'},
            'node_2': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'},
            'node_3': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'},
            'axpy': {'type': 'Axpy', 'kind': 'op', 'op': 'Axpy'},
            'node_4': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'}}
        edges = [
            ('node_1', 'axpy', {'in': 0, 'out': 0}),
            ('node_2', 'axpy', {'in': 1, 'out': 0}),
            ('node_3', 'axpy', {'in': 2, 'out': 0}),
            ('axpy', 'node_4', {'in': 0, 'out': 0})]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'axpy')
        replacer = AxpyToSSandAdd()
        replacer.replace_op(graph, node)

        scale_node = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'ScaleShift']
        self.assertEqual(len(scale_node), 1)
        add_node = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'Add']
        self.assertEqual(len(add_node), 1)
