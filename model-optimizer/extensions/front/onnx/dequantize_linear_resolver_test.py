"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest
from argparse import Namespace

import numpy as np

from extensions.front.onnx.dequantize_linear_resolver import DequantizeLinearResolver
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes1_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'dequantize': {'kind': 'op', 'op': 'DequantizeLinear'},
    'scale_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_ref_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'cast': {'kind': 'op', 'op': 'Cast', 'type': 'Convert'},
    'shape': {'kind': 'op', 'op': 'ShapeOf', 'type': 'ShapeOf'},
    'bc1': {'kind': 'op', 'op': 'Broadcast', 'type': 'Broadcast'},
    'bc2': {'kind': 'op', 'op': 'Broadcast', 'type': 'Broadcast'},
    'sub': {'kind': 'op', 'op': 'Sub', 'type': 'Subtract'},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'scale_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'axis1': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'axis2': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}


class TestDequantizeLinearResolver(unittest.TestCase):

    def test_dequantize(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'dequantize'),
                             ('scale_param_dq', 'dequantize'),
                             ('zerop_param_dq', 'dequantize'),
                             ('dequantize', 'out'),
                             ],
                            {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)},
                             }, nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'cast'),
                                 ('cast', 'sub'),
                                 ('zerop_param_dq', 'sub'),
                                 ('sub', 'mul'),
                                 ('scale_param_dq', 'mul'),
                                 ('mul', 'out'),
                                 ],
                                {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize_no_zerop(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'dequantize'),
                             ('scale_param_dq', 'dequantize'),
                             ('dequantize', 'out'),
                             ],
                            {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             }, nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'cast'),
                                 ('cast', 'mul'),
                                 ('scale_param_dq', 'mul'),
                                 ('mul', 'out'),
                                 ],
                                {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize_axis(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'dequantize'),
                             ('scale_param_dq', 'dequantize'),
                             ('zerop_param_dq', 'dequantize'),
                             ('dequantize', 'out'),
                             ],
                            {'dequantize': {'axis': 1},
                             'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)},
                             }, nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'cast', {'out': 0}),
                                 ('input', 'shape', {'out': 0}),
                                 ('cast', 'sub'),
                                 ('zerop_param_dq', 'bc2'),
                                 ('shape', 'bc2', {'out': 0}),
                                 ('axis2', 'bc2'),
                                 ('bc2', 'sub'),
                                 ('sub', 'mul'),
                                 ('scale_param_dq', 'bc1'),
                                 ('shape', 'bc1', {'out': 0}),
                                 ('axis1', 'bc1'),
                                 ('bc1', 'mul'),
                                 ('mul', 'out'),
                                 ],
                                {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'zerop_param_dq': {'shape': np.array([]), 'value': np.uint8(0)},
                                 'axis1': {'shape': np.array([]), 'value': np.int64(1)},
                                 'axis2': {'shape': np.array([]), 'value': np.int64(1)}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize_axis_no_zerop(self):
        graph = build_graph(nodes1_attributes,
                            [('input', 'dequantize'),
                             ('scale_param_dq', 'dequantize'),
                             ('dequantize', 'out'),
                             ],
                            {'dequantize': {'axis': 1},
                             'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                             }, nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'cast', {'out': 0}),
                                 ('input', 'shape', {'out': 0}),
                                 ('cast', 'mul'),
                                 ('scale_param_dq', 'bc1'),
                                 ('shape', 'bc1', {'out': 0}),
                                 ('axis1', 'bc1'),
                                 ('bc1', 'mul'),
                                 ('mul', 'out'),
                                 ],
                                {'scale_param_dq': {'shape': np.array([]), 'value': np.float32(1.0 / 255)},
                                 'axis1': {'shape': np.array([]), 'value': np.int64(1)}
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'
        DequantizeLinearResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)
