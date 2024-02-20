# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.middle.SharedWeightsDuplication import SharedWeightsDuplication
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'const': {'shape': None, 'type': 'Const', 'kind': 'op', 'op': 'Const'},
    # Mul and Add operations
    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_3': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_3_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'op_output': {'op': 'Result', 'kind': 'op'}
}


class DuplicateSharedWeightsTests(unittest.TestCase):
    def test_duplicate_shared_weights_1(self):
        graph = build_graph(nodes_attributes,
                            [('const', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_1_w', 'mul_3'),
                             ('mul_3', 'mul_3_data'),
                             ('mul_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('mul_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data'),
                             ('concat_1_data', 'op_output')
                             ],
                            {'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])}},
                            nodes_with_edges_only=True
                            )

        graph_ref = build_graph(nodes_attributes,
                                [
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_3_w', 'mul_3'),
                                 ('mul_3', 'mul_3_data'),
                                 ('mul_1_data', 'concat_1'),
                                 ('mul_2_data', 'concat_1'),
                                 ('mul_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data'),
                                 ('concat_1_data', 'op_output')
                                ],
                                {'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_3_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 }, nodes_with_edges_only=True)

        SharedWeightsDuplication().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)
