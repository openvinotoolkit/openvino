# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.ThresholdedReluDecomposition import ThresholdedReluDecomposition
from openvino.tools.mo.front.common.partial_infer.utils import float_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const

nodes_attributes = {
    'parameter': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'trelu': {'type': None, 'kind': 'op', 'op': 'ThresholdedRelu', 'alpha': 0.75, 'name': 'my_trelu'},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    'cast': {'type': 'Convert', 'kind': 'op', 'op': 'Cast'},
    'greater': {'type': 'Greater', 'kind': 'op', 'op': 'Greater'},
    'mul': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul', 'name': 'my_trelu'},
    'squeeze2': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    **const('alpha', float_array([0.75])),
}


class ThresholdedReluDecompositionTest(unittest.TestCase):
    def test_trelu(self):
        graph = build_graph(nodes_attributes,
                            [('parameter', 'trelu', {'in': 0, 'out': 0}),
                             ('trelu', 'result', {'in': 0, 'out': 0}),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('parameter', 'greater', {'in': 0, 'out': 0}),
                                 ('alpha', 'greater', {'in': 1, 'out': 0}),
                                 ('greater', 'cast', {'in': 0, 'out': 0}),
                                 ('parameter', 'mul', {'in': 0, 'out': 0}),
                                 ('cast', 'mul', {'in': 1, 'out': 0}),
                                 ('mul', 'result', {'in': 0, 'out': 0}),
                                 ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        ThresholdedReluDecomposition().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(len(graph.get_op_nodes(name='my_trelu')) == 1 and
                        graph.get_op_nodes(name='my_trelu')[0].op == 'Mul')
