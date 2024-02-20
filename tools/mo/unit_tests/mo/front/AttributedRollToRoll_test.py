# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.AttributedRollToRoll import AttributedRollToRoll
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const, result, regular_op

nodes_attributes = {
    **regular_op('placeholder', {'type': 'Parameter'}),
    **regular_op('attr_roll', {'type': 'AttributedRoll', 'op': 'AttributedRoll', 'axes': int64_array([-1, 2, 3]),
                               'shift': int64_array([5, -2, 3])}),
    **result('result'),

    # new Roll node and inputs
    **regular_op('roll', {'type': 'Roll'}),
    **const('roll_axes', int64_array([-1, 2, 3])),
    **const('roll_shift', int64_array([5, -2, 3]))
}


class AttributedRollToRollTest(unittest.TestCase):
    def test_axes_shift(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_roll', {'in': 0, 'out': 0}),
                             ('attr_roll', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'roll', {'in': 0, 'out': 0}),
                                 ('roll_shift', 'roll', {'in': 1, 'out': 0}),
                                 ('roll_axes', 'roll', {'in': 2, 'out': 0}),
                                 ('roll', 'result')], {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedRollToRoll().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Roll')[0]]['name'] == 'attr_roll')

    def test_axes(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_roll', {'in': 0, 'out': 0}),
                             ('attr_roll', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)
        Node(graph, 'attr_roll')['axes'] = None

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'roll', {'in': 0, 'out': 0}),
                                 ('roll_shift', 'roll', {'in': 1, 'out': 0}),
                                 ('roll', 'result')], {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedRollToRoll().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Roll')[0]]['name'] == 'attr_roll')
