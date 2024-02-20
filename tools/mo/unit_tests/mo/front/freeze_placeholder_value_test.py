# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.freeze_placeholder_value import FreezePlaceholderValue
from unit_tests.utils.graph import build_graph

nodes_bool = {
    '0': {'name': 'input1', 'kind': 'op', 'op': 'Parameter', 'data_type': bool, 'shape': np.array([])},
    '1': {'name': 'input2', 'kind': 'op', 'op': 'Parameter', 'data_type': bool, 'shape': np.array([])},
    '2': {'name': 'node_1', 'kind': 'op', 'op': 'NotPlaceholder'},
    '3': {'name': 'node_2', 'kind': 'op', 'op': 'NotPlaceholder'},
    '4': {'name': 'node_3', 'kind': 'op', 'op': 'NotPlaceholder'},
    '5': {'name': 'node_4', 'kind': 'op', 'op': 'NotPlaceholder'},
    '6': {'name': 'output1', 'kind': 'op', 'op': 'Result', 'type': 'Result'},
    '7': {'name': 'output2', 'kind': 'op', 'op': 'Result', 'type': 'Result'}

}
edges = {
    ('0', '2'),
    ('2', '3'),
    ('4', '6'),
    ('1', '5'),
    ('5', '7')
}


class TestFreezePlaceholderValue(unittest.TestCase):
    def test_freeze_true(self):
        graph = build_graph(nodes_bool, edges)
        graph.graph['fw'] = 'tf'
        tested_class = FreezePlaceholderValue()
        graph.graph['freeze_placeholder'] = {'input1': 'True'}
        before_pattern = graph.nodes()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        # number of nodes in the grpaph didn't change
        self.assertEqual(len(before_pattern), len(after_pattern))
        # reach new placeholder
        try:
            new_ph_dict = graph.node[[u for u, v in graph.in_edges('2')][0]]
        except Exception as e:
            self.fail("Can't get frozen placeholder. Broken edge. Additional information: {}".format(e))
        # check value
        self.assertEqual('value' in new_ph_dict, True)
        self.assertEqual(new_ph_dict['value'], True)

    def test_freeze_false(self):
        graph = build_graph(nodes_bool, edges)
        graph.graph['fw'] = 'tf'
        tested_class = FreezePlaceholderValue()
        graph.graph['freeze_placeholder'] = {'input1': 'False'}
        before_pattern = graph.nodes()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        # number of nodes in the grpaph didn't change
        self.assertEqual(len(before_pattern), len(after_pattern))
        # reach new placeholder
        try:
            new_ph_dict = graph.node[[u for u, v in graph.in_edges('2')][0]]
        except Exception as e:
            self.fail("Can't get frozen placeholder. Broken edge. Additional information: {}".format(e))
        # check value
        self.assertEqual('value' in new_ph_dict, True)
        self.assertEqual(new_ph_dict['value'], False)

    def test_freeze_both(self):
        graph = build_graph(nodes_bool, edges)
        graph.graph['fw'] = 'tf'
        tested_class = FreezePlaceholderValue()
        graph.graph['freeze_placeholder'] = {'input1': 'False', 'input2': 'True'}
        before_pattern = graph.nodes()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        # number of nodes in the graph didn't change
        self.assertEqual(len(before_pattern), len(after_pattern))
        # reach new placeholder
        try:
            new_ph_dict_1 = graph.node[[u for u, v in graph.in_edges('2')][0]]
            new_ph_dict_2 = graph.node[[u for u, v in graph.in_edges('5')][0]]
        except Exception as e:
            self.fail("Can't get frozen placeholder. Broken edge. Additional information: {}".format(e))
        # check value
        self.assertEqual('value' in new_ph_dict_1, True)
        self.assertEqual('value' in new_ph_dict_2, True)
        self.assertEqual(new_ph_dict_1['value'], False)
        self.assertEqual(new_ph_dict_2['value'], True)
