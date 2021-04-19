# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np

from extensions.front.onnx.quantize_linear_resolver import QuantizeLinearResolver
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes1_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'quantize': {'kind': 'op', 'op': 'QuantizeLinear', 'axis': 1},
    'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_ref_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'cast': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
    'f_quantize': {'kind': 'op', 'op': 'FakeQuantize', 'type': 'FakeQuantize'},
    'mul1': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul2': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'in_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'in_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}


class TestQuantizeLinearResolver(unittest.TestCase):

    def test_quantize_uint8(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'quantize'),
                             ('scale_param_q', 'quantize'),
                             ('zerop_param_q', 'quantize'),
                             ('quantize', 'out'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_q': {'shape': np.array([]), 'value': np.uint8(128)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'f_quantize'),
                                 ('scale_param_q', 'mul1', {'out': 0}),
                                 ('in_low', 'mul1'),
                                 ('mul1', 'f_quantize'),
                                 ('scale_param_q', 'mul2', {'out': 0}),
                                 ('in_high', 'mul2'),
                                 ('mul2', 'f_quantize'),
                                 ('out_low', 'f_quantize'),
                                 ('out_high', 'f_quantize'),
                                 ('f_quantize', 'cast'),
                                 ('cast', 'out'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': -128},
                                 'in_high': {'shape': np.array([]), 'value': 127},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_int8(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'quantize'),
                             ('scale_param_q', 'quantize'),
                             ('zerop_param_q', 'quantize'),
                             ('quantize', 'out'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_q': {'shape': np.array([]), 'value': np.int8(0)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'f_quantize'),
                                 ('scale_param_q', 'mul1', {'out': 0}),
                                 ('in_low', 'mul1'),
                                 ('mul1', 'f_quantize'),
                                 ('scale_param_q', 'mul2', {'out': 0}),
                                 ('in_high', 'mul2'),
                                 ('mul2', 'f_quantize'),
                                 ('out_low', 'f_quantize'),
                                 ('out_high', 'f_quantize'),
                                 ('f_quantize', 'cast'),
                                 ('cast', 'out'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': -128},
                                 'in_high': {'shape': np.array([]), 'value': 127},
                                 'out_low': {'shape': np.array([]), 'value': -128},
                                 'out_high': {'shape': np.array([]), 'value': 127},
                                 'cast': {'dst_type': np.int8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_no_zerop(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'quantize'),
                             ('scale_param_q', 'quantize'),
                             ('quantize', 'out'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'f_quantize'),
                                 ('scale_param_q', 'mul1', {'out': 0}),
                                 ('in_low', 'mul1'),
                                 ('mul1', 'f_quantize'),
                                 ('scale_param_q', 'mul2', {'out': 0}),
                                 ('in_high', 'mul2'),
                                 ('mul2', 'f_quantize'),
                                 ('out_low', 'f_quantize'),
                                 ('out_high', 'f_quantize'),
                                 ('f_quantize', 'cast'),
                                 ('cast', 'out'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': 0},
                                 'in_high': {'shape': np.array([]), 'value': 255},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_uint8_axis(self):
        nodes2_attributes = {
            'input': {'kind': 'op', 'op': 'AnyOp'},
            'quantize': {'kind': 'op', 'op': 'QuantizeLinear', 'axis': 0},
            'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'zerop_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'out': {'kind': 'op', 'op': 'AnyOp'},
        }

        nodes_ref_attributes2 = {
            'input': {'kind': 'op', 'op': 'AnyOp'},
            'cast': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
            'f_quantize': {'kind': 'op', 'op': 'FakeQuantize', 'type': 'FakeQuantize'},
            'mul1': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
            'mul2': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
            'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'in_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'in_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'out_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'out_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'out': {'kind': 'op', 'op': 'AnyOp'},
            'rank': {'kind': 'op', 'op': 'Rank'},
            'shape': {'kind': 'op', 'op': 'Shape'},
            'range_const0': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'range_const2': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'range': {'kind': 'op', 'op': 'Range', 'type': 'Range'},
            'greater_equal_const': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'greater_equal': {'kind': 'op', 'op': 'GreaterEqual', 'type': 'GreaterEqual'},
            'cast2': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
            'gather': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'scatter_elements_const1': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
            'scatter_elements': {'kind': 'op', 'op': 'ScatterElementsUpdate', 'type': 'ScatterElements'},
            'reshape_low': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
            'reshape_high': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
        }

        graph = build_graph(nodes2_attributes,
                            [('input', 'quantize'),
                             ('scale_param_q', 'quantize'),
                             ('zerop_param_q', 'quantize'),
                             ('quantize', 'out'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_q': {'shape': np.array([]), 'value': np.uint8(128)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes2,
                                [('input', 'f_quantize', {'out': 0}),
                                 ('input', 'rank', {'out': 0}),
                                 ('input', 'shape', {'out': 0}),
                                 ('shape', 'gather'),
                                 ('range_const0', 'range', {'in': 0, 'out': 2}),
                                 ('rank', 'range', {'in': 1, 'out': 0}),
                                 ('range_const2', 'range', {'in': 2, 'out': 0}),
                                 ('range', 'greater_equal', {'in': 0, 'out': 0}),
                                 ('greater_equal_const', 'greater_equal', {'in': 1, 'out': 0}),
                                 ('greater_equal', 'cast2'),
                                 ('cast2', 'scatter_elements', {'in': 0, 'out': 0}),
                                 ('scatter_elements_const1', 'scatter_elements', {'in': 1, 'out': 0}),
                                 ('gather', 'scatter_elements', {'in': 2, 'out': 0}),
                                 ('scale_param_q', 'mul1', {'out': 0}),
                                 ('in_low', 'mul1'),
                                 ('mul1', 'reshape_low', {'in': 0, 'out': 0}),
                                 ('scatter_elements', 'reshape_low', {'in': 1, 'out': 0}),
                                 ('reshape_low', 'f_quantize', {'in': 1, 'out': 0}),
                                 ('scale_param_q', 'mul2', {'out': 0}),
                                 ('in_high', 'mul2'),
                                 ('mul2', 'reshape_high', {'in': 0, 'out': 0}),
                                 ('scatter_elements', 'reshape_high', {'in': 1, 'out': 0}),
                                 ('reshape_high', 'f_quantize', {'in': 2, 'out': 0}),
                                 ('out_low', 'f_quantize', {'in': 3, 'out': 0}),
                                 ('out_high', 'f_quantize', {'in': 4, 'out': 0}),
                                 ('f_quantize', 'cast'),
                                 ('cast', 'out'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': -128},
                                 'in_high': {'shape': np.array([]), 'value': 127},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        #(flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        #self.assertTrue(flag, resp)
