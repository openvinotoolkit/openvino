# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    'sub': {'kind': 'op', 'op': 'Sub', 'type': 'Subtract'},
    'mul': {'kind': 'op', 'op': 'Mul', 'type': 'Multiply'},
    'scale_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
    'zerop_param_dq': {'kind': 'op', 'type': 'Const', 'op': 'Const'},
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
