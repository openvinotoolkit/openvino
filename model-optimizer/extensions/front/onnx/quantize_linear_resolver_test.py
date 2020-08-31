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

from extensions.front.onnx.quantize_linear_resolver import QuantizeLinearResolver
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes1_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'quantize': {'kind': 'op', 'op': 'QuantizeLinear'},
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
