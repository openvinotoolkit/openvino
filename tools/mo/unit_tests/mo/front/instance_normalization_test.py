# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.instance_normalization import InstanceNormalization
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'scale': {'kind': 'op', 'op': 'AnyOp'},
    'B': {'kind': 'op', 'op': 'AnyOp'},
    'node': {'kind': 'op', 'op': 'InstanceNormalization', 'epsilon': None},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}

nodes_ref_attributes = {
    'input': {'kind': 'op', 'op': 'AnyOp'},
    'scale': {'kind': 'op', 'op': 'AnyOp'},
    'B': {'kind': 'op', 'op': 'AnyOp'},
    'start': {'kind': 'op', 'op': 'Const'},
    'step': {'kind': 'op', 'op': 'Const'},
    'rank': {'kind': 'op', 'op': 'Rank'},
    'mvn_axes': {'kind': 'op', 'op': 'Range'},
    'mvn': {'kind': 'op', 'op': 'MVN', 'name': 'node/Ins_Norm/MVN_', 'eps': None},
    'mul': {'kind': 'op', 'op': 'Mul', 'name': 'node/Ins_Norm/mul_'},
    'add': {'kind': 'op', 'op': 'Add', 'name': 'node/Ins_Norm/add_'},
    'out': {'kind': 'op', 'op': 'AnyOp'},
}


class TestInstanceNormalization(unittest.TestCase):
    def test_instance_normalization_test_1(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'node'),
                             ('scale', 'node'),
                             ('B', 'node'),
                             ('node', 'out')
                             ],
                            {'node': {'epsilon': 0.123},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'mvn', {'out': 0}),
                                 ('input', 'rank', {'out': 0}),
                                 ('start', 'mvn_axes'),
                                 ('rank', 'mvn_axes'),
                                 ('step', 'mvn_axes'),
                                 ('mvn_axes', 'mvn'),
                                 ('mvn', 'mul'),
                                 ('scale', 'mul'),
                                 ('mul', 'add'),
                                 ('B', 'add'),
                                 ('add', 'out')
                                 ],
                                {'mvn': {'eps': 0.123, 'eps_mode': 'inside_sqrt', 'normalize_variance': 1},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = InstanceNormalization()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=False)
        self.assertTrue(flag, resp)

    def test_instance_normalization_test_2(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'out', {'out': 0, 'in': 0}),
                             ('input', 'node', {'out': 1}),
                             ('scale', 'node'),
                             ('B', 'node'),
                             ('node', 'out', {'in': 1})
                             ],
                            {'node': {'epsilon': 0.123},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_ref_attributes,
                                [('input', 'out', {'out': 0, 'in': 0}),
                                 ('input', 'mvn', {'out': 1}),
                                 ('input', 'rank', {'out': 1}),
                                 ('start', 'mvn_axes'),
                                 ('rank', 'mvn_axes'),
                                 ('step', 'mvn_axes'),
                                 ('mvn_axes', 'mvn'),
                                 ('mvn', 'mul'),
                                 ('scale', 'mul'),
                                 ('mul', 'add'),
                                 ('B', 'add'),
                                 ('add', 'out', {'in': 1})
                                 ],
                                {'mvn': {'eps': 0.123, 'eps_mode': 'inside_sqrt', 'normalize_variance': 1},
                                 }, nodes_with_edges_only=True)

        graph.stage = 'front'

        tested_class = InstanceNormalization()
        tested_class.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out', check_op_attrs=False)
        self.assertTrue(flag, resp)
