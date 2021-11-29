# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.RollWithEmptyAxesReplacer import RollWithEmptyAxesReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const, result, regular_op

nodes_attributes = {
    **regular_op('placeholder', {'type': 'Parameter'}),
    **regular_op('roll', {'type': 'Roll', 'op': 'Roll', 'axes': int64_array([-1, 2, 3]), 'shift': int64_array([5, -2, 3])}),
    **const('roll_shift', int64_array([5, -2, 3])),
    **result('result'),

    **regular_op('shape_of', {'type': 'ShapeOf'}),
    **regular_op('reshape1', {'type': 'Reshape'}),
    **regular_op('new_roll', {'type': 'Roll'}),
    **regular_op('reshape2', {'type': 'Reshape'}),

    **const('min_one_const', int64_array([-1])),
    **const('zero_const', int64_array([0]))
}


class RollWithEmptyAxesReplacerTest(unittest.TestCase):
    def test_transform(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'roll', {'in': 0, 'out': 0}),
                             ('roll_shift', 'roll', {'in': 1, 'out': 0}),
                             ('roll', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)
        Node(graph, 'roll').add_input_port(2)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'reshape1', {'in': 0, 'out': 0}),
                                 ('min_one_const', 'reshape1', {'in': 1, 'out': 0}),
                                 ('reshape1', 'new_roll', {'in': 0, 'out': 0}),
                                 ('roll_shift', 'new_roll', {'in': 1, 'out': 0}),
                                 ('zero_const', 'new_roll', {'in': 2, 'out': 0}),
                                 ('new_roll', 'reshape2', {'in': 0, 'out': 0}),
                                 ('placeholder', 'shape_of', {'in': 0, 'out': 0}),
                                 ('shape_of', 'reshape2', {'in': 1, 'out': 0}),
                                 ('reshape2', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)

        graph.stage = 'front'

        RollWithEmptyAxesReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

        shape_of_nodes = graph.get_op_nodes(type='ShapeOf')
        self.assertTrue(len(shape_of_nodes) == 1)
        shape_of = shape_of_nodes[0]
        self.assertTrue(shape_of.in_node().soft_get('name') == 'placeholder')
