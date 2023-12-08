# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.dequantize_linear_resolver import DequantizeLinearResolver
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph
import pytest

nodes1_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'input_data': {'kind': 'data', 'shape': None},
    'dequantize': {'kind': 'op', 'op': 'DequantizeLinear', 'axis': 1},
    'dequantize_data': {'kind': 'data', 'shape': None},
    'scale_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'scale_param_dq_data': {'kind': 'data', 'shape': None},
    'zerop_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_dq_data': {'kind': 'data', 'shape': None},
    'out': {'kind': 'op', 'op': 'AnyOp'},
    'out_data': {'kind': 'data', 'shape': None},
    'result': {'kind': 'op', 'op': 'Result'},
}

nodes_ref_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'input_data': {'kind': 'data', 'shape': None},
    'cast': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
    'cast_data': {'kind': 'data', 'shape': None},
    'sub': {'kind': 'op', 'op': 'Sub', 'type': 'Subtract'},
    'sub_data': {'kind': 'data', 'shape': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'mul_data': {'kind': 'data', 'shape': None},
    'scale_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'scale_param_dq_data': {'kind': 'data', 'shape': None},
    'zerop_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_dq_data': {'kind': 'data', 'shape': None},
    'out': {'kind': 'op', 'op': 'AnyOp'},
    'out_data': {'kind': 'data', 'shape': None},
    'result': {'kind': 'op', 'op': 'Result'},

    'sub_reshape_const': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'sub_reshape_const_data': {'kind': 'data', 'shape': None},
    'sub_reshape': {'kind': 'op', 'type': 'Reshape', 'op': 'Reshape'},
    'sub_reshape_data': {'kind': 'data', 'shape': None},

    'mul_reshape_const': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'mul_reshape_const_data': {'kind': 'data', 'shape': None},
    'mul_reshape': {'kind': 'op', 'type': 'Reshape', 'op': 'Reshape'},
    'mul_reshape_data': {'kind': 'data', 'shape': None},
}


class TestDequantizeLinearResolver(unittest.TestCase):

    def test_dequantize(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'dequantize'),
                             ('dequantize', 'dequantize_data'),
                             ('scale_param_dq', 'scale_param_dq_data'),
                             ('zerop_param_dq', 'zerop_param_dq_data'),
                             ('scale_param_dq_data', 'dequantize'),
                             ('zerop_param_dq_data', 'dequantize'),
                             ('dequantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {'input_data': {'shape': int64_array([1, 3, 224, 224])},
                             'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'scale_param_dq_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)},
                             'zerop_param_dq_data': {'shape': np.array([]), 'value': np.uint8(0)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'sub'),
                                 ('zerop_param_dq', 'zerop_param_dq_data'),
                                 ('zerop_param_dq_data', 'sub'),
                                 ('sub', 'sub_data'),
                                 ('sub_data', 'mul'),
                                 ('scale_param_dq', 'scale_param_dq_data'),
                                 ('scale_param_dq_data', 'mul'),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'input_data': {'shape': int64_array([1, 3, 224, 224])},
                                 'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'scale_param_dq_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)},
                                 'zerop_param_dq_data': {'shape': np.array([]), 'value': np.uint8(0)},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize_no_zerop(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'dequantize'),
                             ('dequantize', 'dequantize_data'),
                             ('scale_param_dq', 'scale_param_dq_data'),
                             ('scale_param_dq_data', 'dequantize'),
                             ('dequantize', 'dequantize_data'),
                             ('dequantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {'input_data': {'shape': int64_array([1, 3, 224, 224])},
                             'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'scale_param_dq_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'mul'),
                                 ('scale_param_dq', 'scale_param_dq_data'),
                                 ('scale_param_dq_data', 'mul'),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'input_data': {'shape': int64_array([1, 3, 224, 224])},
                                 'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'scale_param_dq_data': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

class TestDequantizeWithAxis():
    @pytest.mark.parametrize("input_shape, scale_param_value, zero_param_value, target_shape, axis",
                             [(int64_array([1, 3, 4, 4]), np.array([2, 3, 4, 5], dtype=np.float32),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 4, 1]), 2),
                (int64_array([1, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 3, 1, 1]), 1),
                (int64_array([2, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([2, 1, 1, 1]), 0),
                (int64_array([1, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 4, 1]), -2),
                (int64_array([1, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.uint8), int64_array([1, 1, 1, 4]), -1),
                (int64_array([1, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.int32), int64_array([1, 1, 4, 1]), 2),
                (int64_array([1, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.int32), int64_array([1, 3, 1, 1]), 1),
                (int64_array([2, 3, 4, 4]), int64_array([2, 3, 4, 5]),
                 np.array([2, 3, 4, 5], dtype=np.int32), int64_array([2, 1, 1, 1]), 0),
                ])
    def test_dequantize_with_axis(self, input_shape, scale_param_value, zero_param_value, target_shape, axis):
        graph = build_graph(nodes1_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'dequantize'),
                             ('dequantize', 'dequantize_data'),
                             ('scale_param_dq', 'scale_param_dq_data'),
                             ('zerop_param_dq', 'zerop_param_dq_data'),
                             ('scale_param_dq_data', 'dequantize'),
                             ('zerop_param_dq_data', 'dequantize'),
                             ('dequantize_data', 'out'),
                             ('out', 'out_data'),
                             ('out_data', 'result'),
                             ],
                            {'input_data': {'shape': input_shape},
                             'dequantize': {'axis': axis},
                             'scale_param_dq': {'shape': scale_param_value.shape,
                                                'value': scale_param_value},
                             'scale_param_dq_data': {'shape': scale_param_value.shape,
                                                     'value': scale_param_value},
                             'zerop_param_dq': {'shape': zero_param_value.shape,
                                                'value': zero_param_value},
                             'zerop_param_dq_data': {'shape': zero_param_value.shape,
                                                     'value': zero_param_value},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'cast'),
                                 ('cast', 'cast_data'),
                                 ('cast_data', 'sub'),
                                 ('zerop_param_dq', 'zerop_param_dq_data'),

                                 ('zerop_param_dq_data', 'sub_reshape'),
                                 ('sub_reshape_const', 'sub_reshape_const_data'),
                                 ('sub_reshape_const_data', 'sub_reshape'),
                                 ('sub_reshape', 'sub_reshape_data'),
                                 ('sub_reshape_data', 'sub'),

                                 ('sub', 'sub_data'),
                                 ('sub_data', 'mul'),
                                 ('scale_param_dq', 'scale_param_dq_data'),

                                 ('scale_param_dq_data', 'mul_reshape'),
                                 ('mul_reshape_const', 'mul_reshape_const_data'),
                                 ('mul_reshape_const_data', 'mul_reshape'),
                                 ('mul_reshape', 'mul_reshape_data'),
                                 ('mul_reshape_data', 'mul'),

                                 ('mul', 'mul_data'),
                                 ('mul_data', 'out'),
                                 ('out', 'out_data'),
                                 ('out_data', 'result'),
                                 ],
                                {'input_data': {'shape': input_shape},
                                 'scale_param_dq': {'shape': scale_param_value.shape,
                                                    'value': scale_param_value},
                                 'scale_param_dq_data': {'shape': scale_param_value.shape,
                                                         'value': scale_param_value},
                                 'zerop_param_dq': {'shape': zero_param_value.shape,
                                                    'value': zero_param_value},
                                 'zerop_param_dq_data': {'shape': zero_param_value.shape,
                                                         'value': zero_param_value},
                                 'sub_reshape_const_data': {'shape': target_shape.shape, 'value': target_shape},
                                 'mul_reshape_const_data': {'shape': target_shape.shape, 'value': target_shape},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'middle'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        assert flag, resp
