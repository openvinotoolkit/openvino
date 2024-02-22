# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.ReluQuantizeFuse import ReluQuantizeFuse, ReluFakeQuantizeMark
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes = {
    # input
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_d': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Relu
    'relu': {'kind': 'op', 'op': 'ReLU'},
    'relu_d': {'value': None, 'shape': None, 'kind': 'data'},

    # Quantize
    'const_1': {'op': 'Const', 'kind': 'op'},
    'const_1_d': {'kind': 'data', 'value': None},
    'const_2': {'op': 'Const', 'kind': 'op'},
    'const_2_d': {'kind': 'data', 'value': None},
    'const_3': {'op': 'Const', 'kind': 'op'},
    'const_3_d': {'kind': 'data', 'value': None},
    'const_4': {'op': 'Const', 'kind': 'op'},
    'const_4_d': {'kind': 'data', 'value': None},

    'quantize': {'kind': 'op', 'op': 'FakeQuantize'},
    'quantize_d': {'value': None, 'shape': None, 'kind': 'data'},

    'quantize_1': {'kind': 'op', 'op': 'FakeQuantize'},
    'quantize_1_d': {'value': None, 'shape': None, 'kind': 'data'},

    # Result
    'output': {'kind': 'op', 'op': 'Result'},
    'output_1': {'kind': 'op', 'op': 'Result'},

    # Ops for extra connection expressing
    'extra_op': {'kind': 'op', 'op': 'SomeOp'},
    'extra_data': {'kind': 'data'},
}

i8_edges = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('relu', 'relu_d', {'out': 0}),
    ('const_1', 'const_1_d'),
    ('const_2', 'const_2_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),

    # data to op connections
    ('placeholder_d', 'relu'),
    ('relu_d', 'quantize', {'in': 0}),
    ('const_1_d', 'quantize', {'in': 1}),
    ('const_2_d', 'quantize', {'in': 2}),
    ('const_3_d', 'quantize', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('quantize_d', 'output'),
]

ref_i8_edges = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('const_1', 'const_1_d'),
    ('const_2', 'const_2_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),

    # data to op connections
    ('placeholder_d', 'quantize', {'in': 0}),
    ('const_1_d', 'quantize', {'in': 1}),
    ('const_2_d', 'quantize', {'in': 2}),
    ('const_3_d', 'quantize', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('quantize_d', 'output'),

    ('placeholder_d', 'relu', {'out': 0}),
    ('relu', 'relu_d', {'out': 0}),
]

i1_edges = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('relu', 'relu_d', {'out': 0}),
    ('const_1', 'const_1_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),

    # data to op connections
    ('placeholder_d', 'relu'),
    ('relu_d', 'quantize', {'in': 0}),
    ('const_1_d', 'quantize', {'in': 1}),
    ('const_1_d', 'quantize', {'in': 2}),
    ('const_3_d', 'quantize', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('quantize_d', 'output'),
]

ref_i1_edges = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('const_1', 'const_1_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),

    # data to op connections
    ('placeholder_d', 'quantize', {'in': 0}),
    ('const_1_d', 'quantize', {'in': 1}),
    ('const_1_d', 'quantize', {'in': 2}),
    ('const_3_d', 'quantize', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('quantize_d', 'output'),

    ('placeholder_d', 'relu', {'out': 0}),
    ('relu', 'relu_d', {'out': 0}),

]

relu_extra_output = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('relu', 'relu_d', {'out': 0}),

    ('const_1', 'const_1_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),

    # data to op connections
    ('placeholder_d', 'relu'),
    ('relu_d', 'quantize', {'in': 0}),
    ('const_1_d', 'quantize', {'in': 1}),
    ('const_1_d', 'quantize', {'in': 2}),
    ('const_3_d', 'quantize', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('quantize_d', 'output'),

    # extra output of relu
    ('relu_d', 'extra_op'),
    ('extra_op', 'extra_data'),
    ('extra_data', 'output_1'),
]

const_extra = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('relu', 'relu_d', {'out': 0}),

    ('const_1', 'const_1_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),
    ('quantize_1', 'quantize_1_d'),

    # data to op connections
    ('placeholder_d', 'relu', {'out': 0}),
    ('relu_d', 'quantize', {'in': 0}),
    ('relu_d', 'quantize_1', {'in': 0}),

    ('const_1_d', 'quantize', {'in': 1}),
    ('const_1_d', 'quantize', {'in': 2}),
    ('const_1_d', 'quantize_1', {'in': 1}),
    ('const_1_d', 'quantize_1', {'in': 2}),

    ('const_3_d', 'quantize', {'in': 3}),
    ('const_3_d', 'quantize_1', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('const_4_d', 'quantize_1', {'in': 4}),
    ('quantize_d', 'output'),
    ('quantize_1_d', 'output_1'),
]

ref_const_extra = [
    # op to data connections
    ('placeholder', 'placeholder_d'),
    ('const_1', 'const_1_d'),
    ('const_2', 'const_2_d'),
    ('const_3', 'const_3_d'),
    ('const_4', 'const_4_d'),
    ('quantize', 'quantize_d'),
    ('quantize_1', 'quantize_1_d'),

    # data to op connections
    ('placeholder_d', 'quantize', {'in': 0, 'out': 0}),
    ('placeholder_d', 'quantize_1', {'in': 0, 'out': 0}),

    ('const_1_d', 'quantize', {'out': 0, 'in': 1}),
    ('const_1_d', 'quantize', {'out': 0, 'in': 2}),
    ('const_2_d', 'quantize_1', {'out': 0, 'in': 1}),
    ('const_2_d', 'quantize_1', {'out': 0, 'in': 2}),

    ('const_3_d', 'quantize', {'in': 3}),
    ('const_3_d', 'quantize_1', {'in': 3}),
    ('const_4_d', 'quantize', {'in': 4}),
    ('const_4_d', 'quantize_1', {'in': 4}),
    ('quantize_d', 'output'),
    ('quantize_1_d', 'output_1'),

    ('placeholder_d', 'relu', {'out': 0}),
    ('relu', 'relu_d', {'out': 0}),
]


class ReluQuantizeFuseTests(unittest.TestCase):
    def test_classic_i8_positive_case(self):
        graph = build_graph(nodes, i8_edges,
                            {'const_1_d': {'value': np.zeros([1, 2, 3, 4])}, 'quantize': {'levels': 256}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, ref_i8_edges, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_classic_i8_negative_case(self):
        graph = build_graph(nodes, i8_edges,
                            {'const_1_d': {'value': np.full([1, 2, 3, 4], -1)}, 'quantize': {'levels': 256}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, i8_edges, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_classic_i1_positive_case(self):
        graph = build_graph(nodes, i1_edges,
                            {'const_1_d': {'value': np.zeros([1, 2, 3, 4], dtype=np.float32)},
                             'quantize': {'levels': 2}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, ref_i1_edges, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_classic_i1_negative_case(self):
        graph = build_graph(nodes, i1_edges,
                            {'const_1_d': {'value': np.full([1, 2, 3, 4], -1, dtype=np.float32)},
                             'quantize': {'levels': 2}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, ref_i1_edges, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        np.array_equal(np.full([1, 2, 3, 4], float('-inf'), dtype=np.float32), graph_ref.node['const_1_d']['value'])

    def test_relu_extra_outputs_i1_case(self):
        graph = build_graph(nodes, relu_extra_output,
                            {'const_1_d': {'value': np.full([1, 2, 3, 4], -1, dtype=np.float32)},
                             'quantize': {'levels': 2}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, relu_extra_output, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'relu', check_op_attrs=True)
        self.assertTrue(flag, resp)
        np.array_equal(np.full([1, 2, 3, 4], float('-inf'), dtype=np.float32), graph_ref.node['const_1_d']['value'])

    def test_const_extra_outputs_i1_case(self):
        graph = build_graph(nodes, const_extra,
                            {'const_1_d': {'value': np.full([1, 2, 3, 4], -1, dtype=np.float32)},
                             'quantize': {'levels': 2}, 'quantize_1': {'levels': 2}},
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NHWC'
        graph.stage = 'middle'

        graph_ref = build_graph(nodes, ref_const_extra, nodes_with_edges_only=True)

        ReluFakeQuantizeMark().find_and_replace_pattern(graph)
        ReluQuantizeFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'relu', check_op_attrs=True)
        self.assertTrue(flag, resp)
        np.array_equal(np.full([1, 2, 3, 4], float('-inf'), dtype=np.float32), graph_ref.node['const_1_d']['value'])
