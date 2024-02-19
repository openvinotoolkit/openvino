# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.tf.concat import Concat
from unit_tests.utils.graph import build_graph_with_edge_attrs


class TestConcatEdgesReshuffler(unittest.TestCase):
    def test_concat_edges_reshaffle(self):
        graph = build_graph_with_edge_attrs(
            {'axis': {},
             'input_1': {},
             'input_2': {},
             'input_3': {},
             'concat': {'op': 'Concat', 'simple_concat': True, 'axis': 1},
             },
            [('axis', 'concat', {'in': 0}),
             ('input_1', 'concat', {'in': 1}),
             ('input_2', 'concat', {'in': 2}),
             ('input_3', 'concat', {'in': 3})],
        )
        Concat().find_and_replace_pattern(graph=graph)
        for u, v, attrs in graph.in_edges('concat', data=True):
            if attrs['in'] == 0:
                self.assertEqual(u, 'input_1')
            if attrs['in'] == 1:
                self.assertEqual(u, 'input_2')
            if attrs['in'] == 2:
                self.assertEqual(u, 'input_3')
            if attrs['in'] == 3:
                self.assertEqual(u, 'axis')
        self.assertTrue('axis' not in graph.node['concat'])
