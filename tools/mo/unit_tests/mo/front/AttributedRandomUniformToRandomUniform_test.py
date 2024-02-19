# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.AttributedRandomUniformToRandomUniform import AttributedRandomUniformToRandomUniform
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const, result, regular_op

nodes = {
    **regular_op('placeholder', {'type': 'Parameter'}),
    **regular_op('attr_random_uniform', {'type': 'AttributedRandomUniform', 'op': 'AttributedRandomUniform',
                                         'output_type': np.float32,
                                         'min_val': float32_array([-1.5]), 'max_val': float32_array([10.7]),
                                         'shape': int64_array([5, 4, 3])}),
    **result('result'),

    # new RandomUniform node and inputs
    **regular_op('random_uniform', {'type': 'RandomUniform'}),
    **const('min_val', float32_array([-1.5])),
    **const('max_val', float32_array([10.7])),
    **const('shape', int64_array([5, 4, 3])),
}


class AttributedRandomUniformToRandomUniformTest(unittest.TestCase):
    def test_min_max(self):
        graph = build_graph(nodes,
                            [('placeholder', 'attr_random_uniform', {'in': 0, 'out': 0}),
                             ('attr_random_uniform', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes,
                                [('placeholder', 'random_uniform', {'in': 0, 'out': 0}),
                                 ('min_val', 'random_uniform', {'in': 1, 'out': 0}),
                                 ('max_val', 'random_uniform', {'in': 2, 'out': 0}),
                                 ('random_uniform', 'result')], {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedRandomUniformToRandomUniform().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(
            graph.node[graph.get_nodes_with_attributes(op='RandomUniform')[0]]['name'] == 'attr_random_uniform')

    def test_min_max_shape(self):
        graph = build_graph(nodes,
                            [('attr_random_uniform', 'result', {'in': 0, 'out': 0})], {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes,
                                [('shape', 'random_uniform', {'in': 0, 'out': 0}),
                                 ('min_val', 'random_uniform', {'in': 1, 'out': 0}),
                                 ('max_val', 'random_uniform', {'in': 2, 'out': 0}),
                                 ('random_uniform', 'result')], {}, nodes_with_edges_only=True)
        graph.stage = 'front'

        AttributedRandomUniformToRandomUniform().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(
            graph.node[graph.get_nodes_with_attributes(op='RandomUniform')[0]]['name'] == 'attr_random_uniform')
