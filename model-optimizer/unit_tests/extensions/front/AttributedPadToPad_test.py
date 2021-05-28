# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.front.AttributedPadToPad import AttributedPadToPad
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'attr_pad': {'type': None, 'kind': 'op', 'op': 'AttributedPad', 'mode': 'constant', 'name': 'attr_pad',
                 'pads': int64_array([1, 2, 3, 4, 5, 6]).reshape([3, 2]), 'fill_value': 0.75},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new Pad layer and inputs
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    'convert_like': {'type': 'ConvertLike', 'kind': 'op', 'op': 'ConvertLike'},
    **const('pad_begin', int64_array([1, 3, 5])),
    **const('pad_end', int64_array([2, 4, 6])),
    **const('pad_fill', np.array(0.75)),
}


class AttributedPadToPadTest(unittest.TestCase):
    def test_mode_constant(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_pad', {'in': 0, 'out': 0}),
                             ('attr_pad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'pad', {'in': 0, 'out': 0}),
                                 ('pad_begin', 'pad', {'in': 1, 'out': 0}),
                                 ('pad_end', 'pad', {'in': 2, 'out': 0}),
                                 ('pad_fill', 'convert_like', {'in': 0, 'out': 0}),
                                 ('placeholder', 'convert_like', {'in': 1, 'out': 0}),
                                 ('convert_like', 'pad', {'in': 3, 'out': 0}),
                                 ('pad', 'result')
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AttributedPadToPad()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'attr_pad')

    def test_mode_non_constant(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_pad', {'in': 0, 'out': 0}),
                             ('attr_pad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {'attr_pad': {'mode': 'reflect'}}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'pad', {'in': 0, 'out': 0}),
                                 ('pad_begin', 'pad', {'in': 1, 'out': 0}),
                                 ('pad_end', 'pad', {'in': 2, 'out': 0}),
                                 ('pad', 'result')
                                 ],
                                {'pad': {'mode': 'reflect'}}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AttributedPadToPad()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'attr_pad')
