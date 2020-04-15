"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from extensions.middle.quantize_fuses import FakeQuantizeFuse
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes = {
    'placeholder': {'kind': 'op', 'op': 'Placeholder'},
    'placeholder_d': {'kind': 'data', 'shape':  np.array([1, 3, 224, 224]), 'value': None},

    'mi_i': {'kind': 'op', 'op': 'Const'},
    'mi_i_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224]), 'value': None},
    'ma_i': {'kind': 'op', 'op': 'Const'},
    'ma_i_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224]), 'value': None},
    'mi_o': {'kind': 'op', 'op': 'Const'},
    'mi_o_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224]), 'value': None},
    'ma_o': {'kind': 'op', 'op': 'Const'},
    'ma_o_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224]), 'value': None},

    'quantize': {'kind': 'op', 'op': 'FakeQuantize', 'keep_in_IR': True},
    'quantize_d': {'kind': 'data', 'shape': None},

    'mul_val': {'kind': 'op', 'op': 'Const'},
    'mul_val_d': {'kind': 'data', 'shape': np.array([1]), 'value': np.array([5])},

    'mul': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224])},
    'mul_1': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_1_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224])},
    'mul_2': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_2_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224])},
    'mul_3': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_3_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224])},
    'mul_4': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
    'mul_4_d': {'kind': 'data', 'shape': np.array([1, 3, 224, 224])},

    'output': {'kind': 'op', 'op': 'Result'},
}

edges = [
    ('placeholder', 'placeholder_d'),
    ('mi_i', 'mi_i_d'),
    ('ma_i', 'ma_i_d'),
    ('mi_o', 'mi_o_d'),
    ('ma_o', 'ma_o_d'),
    ('quantize', 'quantize_d'),
    ('mul', 'mul_d'),
    ('mul_val', 'mul_val_d'),

    ('placeholder_d', 'quantize', {'in': 0}),
    ('mi_i_d', 'quantize', {'in': 1}),
    ('ma_i_d', 'quantize', {'in': 2}),
    ('mi_o_d', 'quantize', {'in': 3}),
    ('ma_o_d', 'quantize', {'in': 4}),

    ('quantize_d', 'mul', {'in': 0}),
    ('mul_val_d', 'mul', {'in': 1}),

    ('mul_d', 'output'),
]

edges_ref_1 = [
    ('placeholder', 'placeholder_d'),
    ('mi_i', 'mi_i_d'),
    ('ma_i', 'ma_i_d'),
    ('mi_o', 'mi_o_d'),
    ('ma_o', 'ma_o_d'),
    ('quantize', 'quantize_d'),
    ('mul', 'mul_d'),
    ('mul_val', 'mul_val_d'),

    ('placeholder_d', 'mul', {'in': 0}),
    ('mul_val_d', 'mul', {'in': 1}),

    ('mul_d', 'quantize', {'in': 0}),
    ('mi_i_d', 'quantize', {'in': 1}),
    ('ma_i_d', 'quantize', {'in': 2}),
    ('mi_o_d', 'quantize', {'in': 3}),
    ('ma_o_d', 'quantize', {'in': 4}),

    ('quantize_d', 'output'),
]

edges_ref_5 = [
    ('placeholder', 'placeholder_d'),
    ('mi_i', 'mi_i_d'),
    ('ma_i', 'ma_i_d'),
    ('mi_o', 'mi_o_d'),
    ('ma_o', 'ma_o_d'),
    ('quantize', 'quantize_d'),
    ('mul', 'mul_d'),
    ('mul_1', 'mul_1_d'),
    ('mul_2', 'mul_2_d'),
    ('mul_3', 'mul_3_d'),
    ('mul_4', 'mul_4_d'),
    ('mul_val', 'mul_val_d'),


    ('placeholder_d', 'mul', {'in': 0}),
    ('mi_i_d', 'mul_1'),
    ('ma_i_d', 'mul_2'),
    ('mi_o_d', 'mul_3'),
    ('ma_o_d', 'mul_4'),

    ('mul_val_d', 'mul', {'in': 1, 'out': 0}),
    ('mul_d', 'quantize', {'in': 0}),

    ('mul_val_d', 'mul_1', {'in': 1, 'out': 0}),
    ('mul_1_d', 'quantize', {'in': 1}),

    ('mul_val_d', 'mul_2', {'in': 1, 'out': 0}),
    ('mul_2_d', 'quantize', {'in': 2}),

    ('mul_val_d', 'mul_3', {'in': 1, 'out': 0}),
    ('mul_3_d', 'quantize', {'in': 3}),

    ('mul_val_d', 'mul_4', {'in': 1, 'out': 0}),
    ('mul_4_d', 'quantize', {'in': 4}),

    ('quantize_d', 'output'),
]


class TestQuantizeFuses(unittest.TestCase):
    def test_pool_1_port_through_quantize(self):
        graph = build_graph(nodes, edges, {'mul': {'fuse_up_to_quantize_ports': [0]}}, nodes_with_edges_only=True)
        graph.stage = 'middle'
        FakeQuantizeFuse().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, edges_ref_1, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_pool_5_ports_through_quantize(self):
        graph = build_graph(nodes, edges, {'mul': {'fuse_up_to_quantize_ports': [0, 1, 2, 3, 4]}},
                            nodes_with_edges_only=True)
        graph.stage = 'middle'
        FakeQuantizeFuse().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, edges_ref_5, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
