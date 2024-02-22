# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.eliminate import mark_output_reachable_nodes, mark_const_producer_nodes
from unit_tests.utils.graph import build_graph

nodes_attributes = {'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'node_4': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'node_5': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'node_6': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
                    'placeholder_1_data_node': {'value': None, 'kind': 'data'},
                    'placeholder_2_data_node': {'value': None, 'kind': 'data'},
                    'data_node_1': {'value': None, 'kind': 'data'},
                    'data_node_2': {'value': None, 'kind': 'data'},
                    'data_node_3': {'value': None, 'kind': 'data'},
                    'data_node_3_2': {'value': None, 'kind': 'data'},
                    'data_node_4': {'value': None, 'kind': 'data'},
                    'data_node_5': {'value': None, 'shape': None, 'kind': 'data'},
                    'data_node_6': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    'op_output_1': {'kind': 'op', 'op': 'Result'},
                    'op_output_2': {'kind': 'op', 'op': 'Result'}
                    }


class TestEliminatePass(unittest.TestCase):
    def test_mark_output_unreachable_nodes(self):
        """
        Checks that all nodes that are unreachable from output nodes are marked correspondingly.
        The graph doesn't contain data nodes yet.
        "node_4" is output.

        placeholder_1->node_1->node_2
              \
               -> node_3->node_4

        :return: None
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'node_1'),
                             ('node_1', 'node_2'),
                             ('placeholder_1', 'node_3'),
                             ('node_3', 'node_4'),
                             ('node_4', 'op_output')
                             ],
                            {'node_4': {}},
                            nodes_with_edges_only=True)
        mark_output_reachable_nodes(graph)

        self.assertListEqual(sorted(['placeholder_1', 'node_3', 'op_output', 'node_4']),
                             sorted(graph.get_nodes_with_attributes(is_output_reachable=True)))
        self.assertListEqual(sorted(['node_1', 'node_2']),
                             sorted(graph.get_nodes_with_attributes(is_output_reachable=False)))

    def test_mark_output_unreachable_nodes_behind_output(self):
        """
        Checks case when unreachable node is 'behind' (i.e. is the child) of the output node.
        The graph doesn't contain data nodes yet.
        "node_2" is output.

        placeholder_1->node_1->node_2->node_3

        :return: None
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'node_1'),
                             ('node_1', 'node_2'),
                             ('node_2', 'node_3'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {}},
                            nodes_with_edges_only=True)
        mark_output_reachable_nodes(graph)

        self.assertListEqual(sorted(['node_1', 'node_2', 'op_output', 'placeholder_1']),
                             sorted(graph.get_nodes_with_attributes(is_output_reachable=True)))
        self.assertFalse(graph.node['node_3']['is_output_reachable'])

    def test_mark_ops_producing_constant_values(self):
        """
        Checks case when operation produces only constant tensors so it could be removed. If the node produces several
        tensors and at least one of them is not constant then we should not mark this node.
        The graph contains data nodes.
        "data_node_2" and "data_node_5" are output.
        "node_3" produces constant tensor "data_node_3" and non-constant tensor "data_node_3_2".
        "node_6" produces constant tensor "data_node_6".
        "node_4" could be eliminated since it gets constant input.

                             node_6->data_node_6->
                                                  \
        placeholder_1->placeholder_1_data_node->node_1->data_node_1->node_2->data_node_2
                                                  /
        node_3->data_node_3->node_4->data_node_4->
           \
            ->data_node_3_2->node_5->data_node_5

        :return: None
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data_node'),
                             ('placeholder_1_data_node', 'node_1'),
                             ('node_1', 'data_node_1'),
                             ('data_node_1', 'node_2'),
                             ('node_2', 'data_node_2'),
                             ('node_3', 'data_node_3'),
                             ('node_3', 'data_node_3_2'),
                             ('node_6', 'data_node_6'),
                             ('data_node_6', 'node_1'),
                             ('data_node_3_2', 'node_5'),
                             ('node_5', 'data_node_5'),
                             ('data_node_3', 'node_4'),
                             ('data_node_4', 'node_1'),
                             ('data_node_2', 'op_output'),
                             ('data_node_5', 'op_output_1')
                             ],
                            {'data_node_2': {},
                             'data_node_5': {},
                             'data_node_3': {'value': np.array(1)},
                             'data_node_6': {'value': np.array(1)}},
                            nodes_with_edges_only=True)
        mark_const_producer_nodes(graph)
        self.assertTrue((graph.node['node_6']['is_const_producer']))
        self.assertListEqual(sorted(['node_1', 'node_2', 'node_3', 'node_5', 'placeholder_1']),
                             sorted(graph.get_nodes_with_attributes(is_const_producer=False, kind='op')))

        graph.clean_up()
        self.assertTrue('node_3' in graph.nodes())
        self.assertTrue('node_4' not in graph.nodes())
        self.assertTrue('node_6' not in graph.nodes())

    def test_undead_nodes_with_constant_inputs(self):
        """
        Checks that if node of 'undead' type has constant inputs it is not removed from the graph.
        :return: None
        """
        pass

    def test_remove_node_from_graph(self):
        """
        Checks case when remove node from graph.
        The graph doesn't contain removed node yet.
        "node_2" should be removed.

        placeholder_1->node_1->node_2->node_3

        :return: None
        """
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'node_1'),
                             ('node_1', 'node_2'),
                             ('node_2', 'node_3')],
                            nodes_with_edges_only=True)
        graph.erase_node(Node(graph, 'node_2'))

        self.assertListEqual(sorted(['placeholder_1', 'node_1', 'node_3']), sorted(graph.nodes()))
