# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.ReplacePNorm import ReplacePNormNodePattern
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class ReplacePNormNodePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            'placeholder': {'kind': 'op', 'op': None},
            'in_node': {'kind': 'data', 'shape': [1, 3500]},
            'pnorm': {'kind': 'op', 'op': 'pnorm', 'group': 10, 'p': 2.0},
            'pnorm_data': {'kind': 'data', 'shape': [1, 350]},
            'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
        }

    def test_pnorm(self):
        graph = build_graph(self.nodes_attributes,
                            [('placeholder', 'in_node'),
                             ('in_node', 'pnorm'),
                             ('pnorm', 'pnorm_data'),
                             ('pnorm_data', 'out_placeholder')])
        ReplacePNormNodePattern().find_and_replace_pattern(graph)

        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': None},
                                 'in_node': {'kind': 'data', 'shape': [1, 3500]},
                                 'pow_const': {'kind': 'op', 'value': 2.0},
                                 'pow_const_d': {'kind': 'data'},
                                 'pow': {'kind': 'op', 'op': 'Pow'},
                                 'pow_data': {'kind': 'data'},
                                 'reshape':  {'kind': 'op', 'op': 'Reshape'},
                                 'reshape_data': {'kind': 'data'},
                                 'const': {'kind': 'op', 'op': 'Const', 'value': [1, 350, 10]},
                                 'const_data': {'kind': 'data'},
                                 'reduce': {'kind': 'op', 'op': 'ReduceSum'},
                                 'reduce_data': {'kind': 'data'},
                                 'const_1': {'kind': 'op', 'op': 'Const', 'value': 2},
                                 'const_data_1': {'kind': 'data'},

                                 'invpow_const': {'kind': 'op', 'value': 0.5},
                                 'invpow_const_d': {'kind': 'data'},
                                 'invpow': {'kind': 'op', 'op': 'Pow'},
                                 'invpow_data': {'kind': 'data'},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'pow', {'in': 0}),
                                    ('pow', 'pow_data'),
                                    ('pow_data', 'reshape', {'in': 0}),
                                    ('reshape', 'reshape_data'),
                                    ('const', 'const_data'),
                                    ('const_data', 'reshape', {'in': 1}),
                                    ('reshape_data', 'reduce', {'in': 0}),
                                    ('const_1', 'const_data_1'),
                                    ('const_data_1', 'reduce', {'in': 1}),
                                    ('reduce', 'reduce_data'),
                                    ('reduce_data', 'invpow', {'in': 0}),
                                    ('invpow', 'invpow_data'),
                                    ('invpow_data', 'out_placeholder'),

                                    ('pow_const', 'pow_const_d'),
                                    ('invpow_const', 'invpow_const_d'),
                                    ('pow_const_d', 'pow', {'in': 1}),
                                    ('invpow_const_d', 'invpow', {'in': 1}),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder', check_op_attrs=True)
        self.assertTrue(flag, resp)
