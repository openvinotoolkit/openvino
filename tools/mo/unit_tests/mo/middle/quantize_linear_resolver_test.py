# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.quantize_linear_resolver import QuantizeLinearResolver
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph
import pytest

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


class TestQuantizeWithAxis():
    @pytest.mark.parametrize("input_shape, scale_param_value, zero_param_value,target_shape, in_low, in_high, out_low, out_high, axis",
                             [(int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 4, 1]),
                 np.array([-2., -3., -4., -5.]), np.array([253., 252., 251., 250.]),
                 0, 255, 2),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 3, 1, 1]),
                 np.array([-2., -3., -4., -5.]), np.array([253., 252., 251., 250.]),
                 0, 255, 1),
                (int64_array([2, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([2, 1, 1, 1]),
                 np.array([-2., -3., -4., -5.]), np.array([253., 252., 251., 250.]),
                 0, 255, 0),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 1, 4]),
                 np.array([-2., -3., -4., -5.]), np.array([253., 252., 251., 250.]),
                 0, 255, -1),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 4, 1]),
                 np.array([-2., -3., -4., -5.]), np.array([253., 252., 251., 250.]),
                 0, 255, -2),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.int8), int64_array([1, 1, 4, 1]),
                 np.array([-130., -131., -132., -133.]), np.array([125., 124., 123., 122.]),
                 -128.0, 127.0, 2),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.int8), int64_array([1, 3, 1, 1]),
                 np.array([-130., -131., -132., -133.]), np.array([125., 124., 123., 122.]),
                 -128.0, 127.0, 1),
                (int64_array([2, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.int8), int64_array([2, 1, 1, 1]),
                 np.array([-130., -131., -132., -133.]), np.array([125., 124., 123., 122.]),
                 -128.0, 127.0, 0),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.int8), int64_array([1, 1, 1, 4]),
                 np.array([-130., -131., -132., -133.]), np.array([125., 124., 123., 122.]),
                 -128.0, 127.0, -1),
                (int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.int8), int64_array([1, 1, 4, 1]),
                 np.array([-130., -131., -132., -133.]), np.array([125., 124., 123., 122.]),
                 -128.0, 127.0, -2),
                ])
    def test_quantize_with_axis(self, input_shape, scale_param_value, zero_param_value,
                                target_shape, in_low, in_high, out_low, out_high, axis):
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
                                'quantize': {'axis': axis},
                                'input': {'shape': input_shape},
                                'input_data': {'shape': input_shape},
                                'scale_param_q': {'shape': scale_param_value.shape, 'value': scale_param_value},
                                'scale_param_q_data': {'shape': scale_param_value.shape, 'value': scale_param_value},
                                'zerop_param_q': {'shape': zero_param_value.shape, 'value': zero_param_value},
                                'zerop_param_q_data': {'shape': zero_param_value.shape, 'value': zero_param_value},
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
                                {'in_low': {'shape': in_low.shape, 'value': in_low},
                                 'in_low_data': {'shape': in_low.shape, 'value': in_low},
                                 'in_high': {'shape': in_high.shape, 'value': in_high},
                                 'in_high_data': {'shape': in_high.shape, 'value': in_high},
                                 'out_low': {'shape': np.array([]), 'value': out_low},
                                 'out_low_data': {'shape': np.array([]), 'value': out_low},
                                 'out_high': {'shape': np.array([]), 'value': out_high},
                                 'out_high_data': {'shape': np.array([]), 'value': out_high},
                                 'cast': {'dst_type': zero_param_value.dtype},
                                 'low_reshape_const_data': {'shape': target_shape.shape, 'value': target_shape},
                                 'high_reshape_const_data': {'shape': target_shape.shape, 'value': target_shape},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        QuantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp
