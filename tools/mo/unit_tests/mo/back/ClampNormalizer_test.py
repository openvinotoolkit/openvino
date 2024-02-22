# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.ClampNormalizer import ClampNormalizer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect


class AttributedClampNormalizerTests(unittest.TestCase):

    def test_2_inputs(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('a_clamp', [1, 3, 20, 20], {'type': None, 'op': 'Clamp'}),
            **regular_op_with_shaped_data('clamp', [1, 3, 20, 20],
                                          {'type': 'Clamp', 'op': 'AttributedClamp', 'min': -3.5, 'max': 3.5}),
            **valued_const_with_data('min', np.array(-3.5)),
            **valued_const_with_data('max', np.array(3.5)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:a_clamp'),
                 *connect('min', '1:a_clamp'),
                 *connect('max', '2:a_clamp'),
                 *connect('a_clamp', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        ClampNormalizer().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, [*connect('placeholder', '0:clamp'), *connect('clamp', 'result')])

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_all_dynamic_inputs(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('min', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('max', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('a_clamp', [1, 3, 20, 20], {'type': None, 'op': 'Clamp'}),
            **regular_op_with_shaped_data('maximum', [1, 3, 20, 20], {'type': 'Maximum', 'op': 'Maximum'}),
            **regular_op_with_shaped_data('minimum', [1, 3, 20, 20], {'type': 'Minimum', 'op': 'Minimum'}),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:a_clamp'),
                 *connect('min', '1:a_clamp'),
                 *connect('max', '2:a_clamp'),
                 *connect('a_clamp', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        ClampNormalizer().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, [*connect('placeholder', '0:maximum'),
                                        *connect('min', '1:maximum'),
                                        *connect('maximum', '0:minimum'),
                                        *connect('max', '1:minimum'),
                                        *connect('minimum', 'result')
                                        ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_no_max_input(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('a_clamp', [1, 3, 20, 20], {'type': None, 'op': 'Clamp'}),
            **regular_op_with_shaped_data('maximum', [1, 3, 20, 20], {'type': 'Maximum', 'op': 'Maximum'}),
            **valued_const_with_data('min', np.array(-3.5)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:a_clamp'),
                 *connect('min', '1:a_clamp'),
                 *connect('a_clamp', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        ClampNormalizer().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, [*connect('placeholder', '0:maximum'),
                                        *connect('min', '1:maximum'),
                                        *connect('maximum', 'result')
                                        ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)

    def test_no_min_input(self):
        nodes = {
            **regular_op_with_shaped_data('placeholder', [1, 3, 20, 20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('a_clamp', [1, 3, 20, 20], {'type': None, 'op': 'Clamp'}),
            **regular_op_with_shaped_data('minimum', [1, 3, 20, 20], {'type': 'Minimum', 'op': 'Minimum'}),
            **valued_const_with_data('max', np.array(3.5)),
            **result('result'),
        }
        edges = [*connect('placeholder', '0:a_clamp'),
                 *connect('max', '2:a_clamp'),
                 *connect('a_clamp', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        ClampNormalizer().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes, [*connect('placeholder', '0:minimum'),
                                        *connect('max', '1:minimum'),
                                        *connect('minimum', 'result')
                                        ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)
