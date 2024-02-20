# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.GatherNormalizer import GatherTreeNormalizer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.passes.eliminate import shape_inference
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect


class GatherTreeNormalizerTests(unittest.TestCase):
    def test_gather_tree_normalizer(self):
        nodes = {
            **regular_op_with_shaped_data('data_0', [100, 1, 10], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('data_1', [100, 1, 10], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('data_2', [1], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('gather_tree', [1], {'type': 'GatherTree'}),
            **valued_const_with_data('const', np.array([2])),
            **result('result'),
        }
        edges = [*connect('data_0', '0:gather_tree'),
                 *connect('data_1', '1:gather_tree'),
                 *connect('data_2', '2:gather_tree'),
                 *connect('const', '3:gather_tree'),
                 *connect('gather_tree', 'result'),
                 ]
        ref_edges = [*connect('data_0', '0:gather_tree'),
                     *connect('data_1', '1:gather_tree'),
                     *connect('data_2', '2:gather_tree'),
                     *connect('const', '0:squeeze'),
                     *connect('squeeze_axis', '1:squeeze'),
                     *connect('squeeze', '3:gather_tree'),
                     *connect('gather_tree', 'result'),]
        ref_nodes = nodes.copy()
        ref_nodes.update({**valued_const_with_data('squeeze_axis', int64_array([0])),
                          **regular_op_with_shaped_data('squeeze', [], {'type': 'Squeeze'})})
        graph = build_graph(nodes, edges)
        GatherTreeNormalizer().find_and_replace_pattern(graph)
        # run shape inference to make sure that shape overriding happened
        shape_inference(graph)

        ref_graph = build_graph(ref_nodes, ref_edges)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)
