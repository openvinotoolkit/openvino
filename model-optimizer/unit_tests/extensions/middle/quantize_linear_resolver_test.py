# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.quantize_linear_resolver import QuantizeLinearResolver
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes1_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'input_data': {'kind': 'data', 'shape': None},
    'quantize': {'kind': 'op', 'op': 'QuantizeLinear', 'axis': 1},
    'quantize_data': {'kind': 'data', 'shape': None},
    'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'scale_param_q_data': {'kind': 'data', 'shape': None},
    'zerop_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_q_data': {'kind': 'data', 'shape': None},
    'out': {'kind': 'op', 'op': 'AnyOp'},
    'out_data': {'kind': 'data', 'shape': None},
    'result': {'kind': 'op', 'op': 'Result'},
}

nodes_ref_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'input_data': {'kind': 'data', 'shape': None},
    'cast': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
    'cast_data': {'kind': 'data', 'shape': None},
    'f_quantize': {'kind': 'op', 'op': 'FakeQuantize', 'type': 'FakeQuantize'},
    'f_quantize_data': {'kind': 'data', 'shape': None},
    'mul1': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul1_data': {'kind': 'data', 'shape': None},
    'mul2': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul2_data': {'kind': 'data', 'shape': None},
    'scale_param_q': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'scale_param_q_data': {'kind': 'data', 'shape': None},
    'in_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'in_low_data': {'kind': 'data', 'shape': None},
    'in_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'in_high_data': {'kind': 'data', 'shape': None},
    'out_low': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out_low_data': {'kind': 'data', 'shape': None},
    'out_high': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out_high_data': {'kind': 'data', 'shape': None},
    'out': {'kind': 'op', 'op': 'AnyOp'},
    'out_data': {'kind': 'data', 'shape': None},
    'result': {'kind': 'op', 'op': 'Result'},

    'high_reshape_const': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'high_reshape_const_data': {'kind': 'data', 'shape': None},
    'high_reshape': {'kind': 'op', 'type': 'Reshape', 'op': 'Reshape'},
    'high_reshape_data': {'kind': 'data', 'shape': None},

    'low_reshape_const': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'low_reshape_const_data': {'kind': 'data', 'shape': None},
    'low_reshape': {'kind': 'op', 'type': 'Reshape', 'op': 'Reshape'},
    'low_reshape_data': {'kind': 'data', 'shape': None},
}


class TestQuantizeLinearResolver(unittest.TestCase):

    def test_quantize_uint8(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'quantize'),
                             ('scale_param_q', 'scale_param_q_data'),
                             ('scale_param_q_data', 'quantize'),
                             ('zerop_param_q', 'zerop_param_q_data'),
                             ('zerop_param_q_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('quantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {
                                'quantize': {'axis': 2},
                                'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                'scale_param_q_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                'zerop_param_q': {'shape': np.array([]), 'value': np.uint8(128)},
                                'zerop_param_q_data': {'shape': np.array([]), 'value': np.uint8(128)},
                            }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'f_quantize'),
                                 ('scale_param_q', 'scale_param_q_data'),
                                 ('scale_param_q_data', 'mul1', {'out': 0}),
                                 ('in_low', 'in_low_data'),
                                 ('in_low_data', 'mul1'),
                                 ('mul1', 'mul1_data'),
                                 ('mul1_data', 'f_quantize'),
                                 ('f_quantize', 'f_quantize_data'),
                                 ('scale_param_q_data', 'mul2', {'out': 0}),
                                 ('in_high', 'in_high_data'),
                                 ('in_high_data', 'mul2'),
                                 ('mul2', 'mul2_data'),
                                 ('mul2_data', 'f_quantize'),
                                 ('out_low', 'out_low_data'),
                                 ('out_low_data', 'f_quantize'),
                                 ('out_high', 'out_high_data'),
                                 ('out_high_data', 'f_quantize'),
                                 ('f_quantize_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': -128},
                                 'in_low_data': {'shape': np.array([]), 'value': -128},
                                 'in_high': {'shape': np.array([]), 'value': 127},
                                 'in_high_data': {'shape': np.array([]), 'value': 127},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_low_data': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'out_high_data': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_uint8_with_axis2(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'quantize'),
                             ('scale_param_q', 'scale_param_q_data'),
                             ('scale_param_q_data', 'quantize'),
                             ('zerop_param_q', 'zerop_param_q_data'),
                             ('zerop_param_q_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('quantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {
                                'quantize': {'axis': 2},
                                'input': {'shape': int64_array([1, 3, 4, 4])},
                                'input_data': {'shape': int64_array([1, 3, 4, 4])},
                                'scale_param_q': {'shape': np.array([4]), 'value': np.array([2, 3, 4, 5])},
                                'scale_param_q_data': {'shape': np.array([4]), 'value': np.array([2, 3, 4, 5])},
                                'zerop_param_q': {'shape': np.array([4]), 'value': np.array([2, 3, 4, 5],
                                                                                            dtype=np.uint8)},
                                'zerop_param_q_data': {'shape': np.array([4]), 'value': np.array([2, 3, 4, 5],
                                                                                                 dtype=np.uint8)},
                            }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'f_quantize'),
                                 ('scale_param_q', 'scale_param_q_data'),
                                 ('scale_param_q_data', 'mul1', {'out': 0}),
                                 ('in_low', 'in_low_data'),
                                 ('in_low_data', 'mul1'),
                                 ('mul1', 'mul1_data'),
                                 ('mul1_data', 'high_reshape'),
                                 ('high_reshape_const', 'high_reshape_const_data'),
                                 ('high_reshape_const_data', 'high_reshape'),
                                 ('high_reshape', 'high_reshape_data'),
                                 ('high_reshape_data', 'f_quantize'),

                                 ('f_quantize', 'f_quantize_data'),
                                 ('scale_param_q_data', 'mul2', {'out': 0}),
                                 ('in_high', 'in_high_data'),
                                 ('in_high_data', 'mul2'),
                                 ('mul2', 'mul2_data'),
                                 ('mul2_data', 'low_reshape'),
                                 ('low_reshape', 'low_reshape_data'),
                                 ('low_reshape_data', 'f_quantize'),
                                 ('low_reshape_const', 'low_reshape_const_data'),
                                 ('low_reshape_const_data', 'low_reshape'),
                                 ('out_low', 'out_low_data'),
                                 ('out_low_data', 'f_quantize'),

                                 ('out_high', 'out_high_data'),
                                 ('out_high_data', 'f_quantize'),
                                 ('f_quantize_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'in_low': {'shape': np.array([4]), 'value': np.array([-2., -3., -4., -5.])},
                                 'in_low_data': {'shape': np.array([4]), 'value': np.array([-2., -3., -4., -5.])},
                                 'in_high': {'shape': np.array([4]), 'value': np.array([253., 252., 251., 250.])},
                                 'in_high_data': {'shape': np.array([4]), 'value': np.array([253., 252., 251., 250.])},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_low_data': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'out_high_data': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_uint8_with_axis0(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'quantize'),
                             ('scale_param_q', 'scale_param_q_data'),
                             ('scale_param_q_data', 'quantize'),
                             ('zerop_param_q', 'zerop_param_q_data'),
                             ('zerop_param_q_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('quantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {
                                'quantize': {'axis': 0},
                                'input': {'shape': int64_array([2, 3, 4, 4])},
                                'input_data': {'shape': int64_array([2, 3, 4, 4])},
                                'scale_param_q': {'shape': np.array([2]), 'value': np.array([2, 3])},
                                'scale_param_q_data': {'shape': np.array([2]), 'value': np.array([2, 3])},
                                'zerop_param_q': {'shape': np.array([2]), 'value': np.array([2, 4],
                                                                                            dtype=np.uint8)},
                                'zerop_param_q_data': {'shape': np.array([2]), 'value': np.array([2, 4],
                                                                                                 dtype=np.uint8)},
                            }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'f_quantize'),
                                 ('scale_param_q', 'scale_param_q_data'),
                                 ('scale_param_q_data', 'mul1', {'out': 0}),
                                 ('in_low', 'in_low_data'),
                                 ('in_low_data', 'mul1'),
                                 ('mul1', 'mul1_data'),
                                 ('mul1_data', 'high_reshape'),
                                 ('high_reshape_const', 'high_reshape_const_data'),
                                 ('high_reshape_const_data', 'high_reshape'),
                                 ('high_reshape', 'high_reshape_data'),
                                 ('high_reshape_data', 'f_quantize'),

                                 ('f_quantize', 'f_quantize_data'),
                                 ('scale_param_q_data', 'mul2', {'out': 0}),
                                 ('in_high', 'in_high_data'),
                                 ('in_high_data', 'mul2'),
                                 ('mul2', 'mul2_data'),
                                 ('mul2_data', 'low_reshape'),
                                 ('low_reshape', 'low_reshape_data'),
                                 ('low_reshape_data', 'f_quantize'),
                                 ('low_reshape_const', 'low_reshape_const_data'),
                                 ('low_reshape_const_data', 'low_reshape'),
                                 ('out_low', 'out_low_data'),
                                 ('out_low_data', 'f_quantize'),

                                 ('out_high', 'out_high_data'),
                                 ('out_high_data', 'f_quantize'),
                                 ('f_quantize_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'in_low': {'shape': np.array([2]), 'value': np.array([-2., -4.])},
                                 'in_low_data': {'shape': np.array([2]), 'value': np.array([-2., -4.])},
                                 'in_high': {'shape': np.array([2]), 'value': np.array([253., 251.])},
                                 'in_high_data': {'shape': np.array([2]), 'value': np.array([253., 251.])},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_low_data': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'out_high_data': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_int8(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'quantize'),
                             ('scale_param_q', 'scale_param_q_data'),
                             ('scale_param_q_data', 'quantize'),
                             ('zerop_param_q', 'zerop_param_q_data'),
                             ('zerop_param_q_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('quantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'scale_param_q_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_q': {'shape': np.array([]), 'value': np.int8(0)},
                             'zerop_param_q_data': {'shape': np.array([]), 'value': np.int8(0)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'f_quantize'),
                                 ('scale_param_q', 'scale_param_q_data'),
                                 ('scale_param_q_data', 'mul1', {'out': 0}),
                                 ('in_low', 'in_low_data'),
                                 ('in_low_data', 'mul1'),
                                 ('mul1', 'mul1_data'),
                                 ('mul1_data', 'f_quantize'),
                                 ('f_quantize', 'f_quantize_data'),
                                 ('scale_param_q_data', 'mul2', {'out': 0}),
                                 ('in_high', 'in_high_data'),
                                 ('in_high_data', 'mul2'),
                                 ('mul2', 'mul2_data'),
                                 ('mul2_data', 'f_quantize'),
                                 ('out_low', 'out_low_data'),
                                 ('out_low_data', 'f_quantize'),
                                 ('out_high', 'out_high_data'),
                                 ('out_high_data', 'f_quantize'),
                                 ('f_quantize_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': -128},
                                 'in_low_data': {'shape': np.array([]), 'value': -128},
                                 'in_high': {'shape': np.array([]), 'value': 127},
                                 'in_high_data': {'shape': np.array([]), 'value': 127},
                                 'out_low': {'shape': np.array([]), 'value': -128},
                                 'out_low_data': {'shape': np.array([]), 'value': -128},
                                 'out_high': {'shape': np.array([]), 'value': 127},
                                 'out_high_data': {'shape': np.array([]), 'value': 127},
                                 'cast': {'dst_type': np.int8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_no_zerop(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('scale_param_q', 'scale_param_q_data'),
                             ('scale_param_q_data', 'quantize'),
                             ('quantize', 'quantize_data'),
                             ('quantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {'scale_param_q': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'scale_param_q_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'f_quantize'),
                                 ('scale_param_q', 'scale_param_q_data'),
                                 ('scale_param_q_data', 'mul1', {'out': 0}),
                                 ('in_low', 'in_low_data'),
                                 ('in_low_data', 'mul1'),
                                 ('mul1', 'mul1_data'),
                                 ('mul1_data', 'f_quantize'),
                                 ('f_quantize', 'f_quantize_data'),
                                 ('scale_param_q_data', 'mul2', {'out': 0}),
                                 ('in_high', 'in_high_data'),
                                 ('in_high_data', 'mul2'),
                                 ('mul2', 'mul2_data'),
                                 ('mul2_data', 'f_quantize'),
                                 ('out_low', 'out_low_data'),
                                 ('out_low_data', 'f_quantize'),
                                 ('out_high', 'out_high_data'),
                                 ('out_high_data', 'f_quantize'),
                                 ('f_quantize_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'in_low': {'shape': np.array([]), 'value': 0},
                                 'in_low_data': {'shape': np.array([]), 'value': 0},
                                 'in_high': {'shape': np.array([]), 'value': 255},
                                 'in_high_data': {'shape': np.array([]), 'value': 255},
                                 'out_low': {'shape': np.array([]), 'value': 0},
                                 'out_low_data': {'shape': np.array([]), 'value': 0},
                                 'out_high': {'shape': np.array([]), 'value': 255},
                                 'out_high_data': {'shape': np.array([]), 'value': 255},
                                 'cast': {'dst_type': np.uint8}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
