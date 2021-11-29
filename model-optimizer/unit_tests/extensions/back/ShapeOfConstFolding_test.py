# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.back.ShapeOfConstFolding import ShapeOfConstFolding
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

const_value = np.random.rand(1, 3, 30, 30)
nodes_attributes = {'input': {'shape': int64_array([1, 3, 30, 30]), 'type': 'Parameter', 'kind': 'op',
                              'op': 'Parameter'},
                    'input_data': {'value': None, 'shape': int64_array([1, 3, 30, 30]), 'kind': 'data'},
                    'const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': const_value},
                    'const_data': {'kind': 'data', 'value': const_value},
                    'shapeof_input': {'kind': 'op', 'op': 'ShapeOf', 'value': int64_array([1, 3, 30, 30])},
                    'shapeof_input_data': {'value': None, 'shape': None, 'kind': 'data',
                                           'value': int64_array([1, 3, 30, 30])},

                    'shapeof_const': {'kind': 'op', 'op': 'ShapeOf', 'value': int64_array([1, 3, 30, 30])},
                    'shapeof_const_data': {'value': None, 'shape': None, 'kind': 'data',
                                           'value': int64_array([1, 3, 30, 30])},

                    'mul': {'kind': 'op', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a * b)},
                    'mul_data': {'kind': 'data', 'value': np.array([1, 9, 900, 900])},
                    'last': {'kind': 'op', 'op': 'Result'},

                    # new nodes
                    'new_const_shapeof': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                          'value': int64_array([1, 3, 30, 30])}
                    }

const_value2 = np.random.rand(30, 30)
nodes_attributes2 = {'input': {'shape': int64_array([1, 3, 30, 30]), 'type': 'Parameter', 'kind': 'op',
                               'op': 'Parameter'},
                     'input_data': {'value': None, 'shape': int64_array([1, 3, 30, 30]), 'kind': 'data'},

                     'const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': const_value2},
                     'const_data': {'kind': 'data', 'value': const_value2},

                     'shapeof_const': {'kind': 'op', 'op': 'ShapeOf', 'value': int64_array([2700, 30])},
                     'shapeof_const_data': {'value': int64_array([2700, 30]), 'shape': None, 'kind': 'data'},

                     'gather': {'kind': 'op', 'op': 'Gather', 'batch_dims': 0},
                     'gather_data': {'kind': 'data'},

                     'const_concat': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': [1]},
                     'const_concat_data': {'kind': 'data', 'value': [1]},
                     'concat': {'kind': 'op', 'op': 'Concat'},
                     'concat_data': {'kind': 'data'},

                     'reshape': {'kind': 'op', 'op': 'Reshape'},
                     'reshape_data': {'kind': 'data'},

                     'matmul': {'kind': 'op', 'op': 'MatMul'},
                     'matmul_data': {'kind': 'data'},
                     'last': {'kind': 'op', 'op': 'Result'},

                     # new nodes
                     'new_const_shapeof': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                          'value': int64_array([2700, 30])},
                     }


class ShapeOfConstFoldingTests(unittest.TestCase):
    def test_const_with_one_output(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'input_data'),
                             ('input_data', 'shapeof_input'),
                             ('shapeof_input', 'shapeof_input_data'),
                             ('shapeof_input_data', 'mul'),
                             ('const', 'const_data'),
                             ('const_data', 'shapeof_const'),
                             ('shapeof_const', 'shapeof_const_data'),
                             ('shapeof_const_data', 'mul'),
                             ('mul', 'mul_data'),
                             ('mul_data', 'last')],
                            {
                                'input': {'shape': int64_array([1, 3, 30, 30])},
                                'input_data': {'shape': int64_array([1, 3, 30, 30])},
                                'shapeof_input': {'value': int64_array([1, 3, 30, 30])},
                                'shapeof_input_data': {'value': int64_array([1, 3, 30, 30])},
                                'const': {'value': const_value},
                                'const_data': {'value': const_value},
                                'shapeof_const': {'value': int64_array([1, 3, 30, 30])},
                                'shapeof_const_data': {'value': int64_array([1, 3, 30, 30])},
                                'mul_data': {'value': int64_array([1, 9, 900, 900])},
                            },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'shapeof_input'),
                                 ('shapeof_input', 'shapeof_input_data'),
                                 ('shapeof_input_data', 'mul'),
                                 ('new_const_shapeof', 'shapeof_const_data'),
                                 ('shapeof_const_data', 'mul'),
                                 ('mul', 'mul_data'),
                                 ('mul_data', 'last')],
                                {
                                    'input': {'shape': int64_array([1, 3, 30, 30])},
                                    'input_data': {'shape': int64_array([1, 3, 30, 30])},
                                    'shapeof_input': {'value': int64_array([1, 3, 30, 30])},
                                    'shapeof_input_data': {'value': int64_array([1, 3, 30, 30])},
                                    'new_const_shapeof': {'value': int64_array([1, 3, 30, 30])},
                                    'shapeof_const_data': {'value': int64_array([1, 3, 30, 30])},
                                    'mul_data': {'value': int64_array([1, 9, 900, 900])},
                                },
                                nodes_with_edges_only=True)
        ShapeOfConstFolding().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    def test_const_with_two_outputs(self):
        graph = build_graph(nodes_attributes2,
                            [('input', 'input_data'),
                             ('input_data', 'reshape'),
                             ('const', 'const_data'),
                             ('const_data', 'shapeof_const'),
                             ('shapeof_const', 'shapeof_const_data'),
                             ('shapeof_const_data', 'gather'),
                             ('gather', 'gather_data'),
                             ('const_concat', 'const_concat_data'),
                             ('const_concat_data', 'concat'),
                             ('gather_data', 'concat'),
                             ('concat', 'reshape'),
                             ('reshape', 'reshape_data'),
                             ('reshape_data', 'matmul'),
                             ('const_data', 'matmul'),
                             ('matmul', 'matmul_data'),
                             ('matmul_data', 'last')
                             ],
                            {
                                'input': {'shape': int64_array([1, 3, 30, 30])},
                                'input_data': {'shape': int64_array([1, 3, 30, 30])},
                                'shapeof_const': {'value': int64_array([2700, 30])},
                                'shapeof_const_data': {'value': int64_array([2700, 30])},
                            },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes2,
                                [('input', 'input_data'),
                                 ('input_data', 'reshape'),
                                 ('new_const_shapeof', 'shapeof_const_data'),
                                 ('shapeof_const_data', 'gather'),
                                 ('gather', 'gather_data'),
                                 ('const_concat', 'const_concat_data'),
                                 ('const_concat_data', 'concat'),
                                 ('gather_data', 'concat'),
                                 ('concat', 'reshape'),
                                 ('reshape', 'reshape_data'),
                                 ('reshape_data', 'matmul'),
                                 ('const', 'const_data'),
                                 ('const_data', 'matmul'),
                                 ('matmul', 'matmul_data'),
                                 ('matmul_data', 'last')],
                                {
                                    'input': {'shape': int64_array([1, 3, 30, 30])},
                                    'input_data': {'shape': int64_array([1, 3, 30, 30])},
                                    'new_const_shapeof': {'value': int64_array([2700, 30])},
                                    'shapeof_const_data': {'value': int64_array([2700, 30])},
                                },
                                nodes_with_edges_only=True)
        ShapeOfConstFolding().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)
