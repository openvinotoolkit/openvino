# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.split_normalizer import SqueezeAxis
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'attr_split': {'type': None, 'kind': 'op', 'op': 'AttributedSplit', 'axis': 0, 'num_splits': 2,
                   'squeeze_axis': True},
    'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2, 'squeeze_axis': True},
    **const('split_axis', int64_array(0)),
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat', 'axis': 0},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    'squeeze1': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    'squeeze2': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    **const('squeeze1_axis', int64_array(0)),
    **const('squeeze2_axis', int64_array(0)),
}


class SqueezeAxisTest(unittest.TestCase):
    def test_attributed(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_split', {'in': 0, 'out': 0}),
                             ('attr_split', 'concat', {'in': 0, 'out': 0}),
                             ('attr_split', 'concat', {'in': 1, 'out': 1}),
                             ('concat', 'result', {'in': 0, 'out': 0}),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'attr_split', {'in': 0, 'out': 0}),
                                 ('attr_split', 'squeeze1', {'in': 0, 'out': 0}),
                                 ('squeeze1_axis', 'squeeze1', {'in': 1, 'out': 0}),
                                 ('attr_split', 'squeeze2', {'in': 0, 'out': 1}),
                                 ('squeeze2_axis', 'squeeze2', {'in': 1, 'out': 0}),
                                 ('squeeze1', 'concat', {'in': 0, 'out': 0}),
                                 ('squeeze2', 'concat', {'in': 1, 'out': 0}),
                                 ('concat', 'result', {'in': 0, 'out': 0}),
                                 ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        SqueezeAxis().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_split(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'split', {'in': 0, 'out': 0}),
                             ('split_axis', 'split', {'in': 1, 'out': 0}),
                             ('split', 'concat', {'in': 0, 'out': 0}),
                             ('split', 'concat', {'in': 1, 'out': 1}),
                             ('concat', 'result', {'in': 0, 'out': 0}),
                             ], nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'split', {'in': 0, 'out': 0}),
                                 ('split_axis', 'split', {'in': 1, 'out': 0}),
                                 ('split', 'squeeze1', {'in': 0, 'out': 0}),
                                 ('split_axis', 'squeeze1', {'in': 1, 'out': 0}),
                                 ('split', 'squeeze2', {'in': 0, 'out': 1}),
                                 ('split_axis', 'squeeze2', {'in': 1, 'out': 0}),
                                 ('squeeze1', 'concat', {'in': 0, 'out': 0}),
                                 ('squeeze2', 'concat', {'in': 1, 'out': 0}),
                                 ('concat', 'result', {'in': 0, 'out': 0}),
                                 ], nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        SqueezeAxis().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
