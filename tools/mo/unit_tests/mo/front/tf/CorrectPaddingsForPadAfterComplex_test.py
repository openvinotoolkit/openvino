# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest
import numpy as np


from openvino.tools.mo.front.tf.CorrectPaddingsForPadAfterComplex import CorrectPaddingsForPadAfterComplex
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const


graph_node_attrs = {
    'placeholder_real': {'shape': int64_array([3, 100, 67]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_imag': {'shape': int64_array([3, 100, 67]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'complex': {'kind': 'op', 'op': 'Complex'},
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    **const('pad_begin', int64_array([1, 3, 5])),
    **const('pad_end', int64_array([2, 4, 6])),
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder_real', 'complex', {'in': 0}),
    ('placeholder_imag', 'complex', {'in': 1}),
    ('complex', 'pad', {'in': 0, 'out': 0}),
    ('pad_begin', 'pad', {'in': 1, 'out': 0}),
    ('pad_end', 'pad', {'in': 2, 'out': 0}),
    ('pad', 'abs'),
    ('abs', 'output'),
]


ref_graph_node_attrs = {
    'placeholder_real': {'shape': int64_array([3, 100, 67]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_imag': {'shape': int64_array([3, 100, 67]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'complex': {'kind': 'op', 'op': 'Complex'},
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    **const('pad_begin', int64_array([1, 3, 5])),
    **const('pad_end', int64_array([2, 4, 6])),
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    **const('additional_pad_begin', int64_array([0])),
    **const('additional_pad_end', int64_array([0])),
    'concat_for_pad_begin': {'kind': 'op', 'op': 'Concat', 'type': 'Concat', 'axis': 0},
    'concat_for_pad_end': {'kind': 'op', 'op': 'Concat', 'type': 'Concat', 'axis': 0},
}

ref_graph_edges = [
    ('placeholder_real', 'complex', {'in': 0}),
    ('placeholder_imag', 'complex', {'in': 1}),
    ('complex', 'pad', {'in': 0, 'out': 0}),
    ('pad_begin', 'concat_for_pad_begin', {'in': 0, 'out': 0}),
    ('additional_pad_begin', 'concat_for_pad_begin', {'in': 1, 'out': 0}),
    ('pad_end', 'concat_for_pad_end', {'in': 0, 'out': 0}),
    ('additional_pad_end', 'concat_for_pad_end', {'in': 1, 'out': 0}),
    ('concat_for_pad_begin', 'pad', {'in': 1, 'out': 0}),
    ('concat_for_pad_end', 'pad', {'in': 2, 'out': 0}),
    ('pad', 'abs'),
    ('abs', 'output'),
]


class CorrectPaddingsForPadAfterComplexTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        graph.stage = 'front'
        CorrectPaddingsForPadAfterComplex().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
