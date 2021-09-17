# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.ReverseTransposeNormalization import ReverseTransposeNormalization
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect


class ReverseTransposeNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            **regular_op_with_shaped_data('placeholder', [1, 10, 20, 3], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('transpose', [3, 20, 10, 1],
                                          {'type': 'Transpose', 'op': 'Transpose', 'reverse_order': True}),
            **result('result'),
        }

        cls.ref_nodes_attributes = {
            **regular_op_with_shaped_data('placeholder', [1, 10, 20, 3], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('transpose', [3, 20, 10, 1],
                                          {'type': 'Transpose', 'op': 'Transpose'}),
            **valued_const_with_data('transpose_order', np.array([3, 2, 1, 0])),
            **result('result'),
        }

    def test_splice(self):
        graph = build_graph(self.nodes_attributes,
                            [*connect('placeholder', '0:transpose'),
                             *connect('transpose', 'result'), ])
        ReverseTransposeNormalization().find_and_replace_pattern(graph)
        graph.clean_up()

        ref_graph = build_graph(self.ref_nodes_attributes,
                                [*connect('placeholder', '0:transpose'),
                                 *connect('transpose_order', '1:transpose'),
                                 *connect('transpose', 'result'), ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
