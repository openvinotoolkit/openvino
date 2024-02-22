# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.global_pooling_to_reduce import GlobalPoolingToReduce
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, result, build_graph_with_edge_attrs, const

nodes = {**regular_op('input', {'type': 'Parameter'}),

         **regular_op('relu', {'type': 'Relu'}),
         **regular_op('pooling', {'type': 'Pooling', 'global_pool': True, 'pool_method': 'avg'}),

         **result('result'),

         **regular_op('rank', {'type': 'Rank'}),
         **regular_op('reduce_mean', {'type': 'ReduceMean'}),
         **regular_op('range', {'type': 'Range'}),
         **const('const_1', int64_array(2)),
         **const('const_2', int64_array(1)),

         }
edges = [('input', 'relu', {'in': 0, 'out': 0}), ('relu', 'pooling', {'in': 0, 'out': 0}),
         ('pooling', 'result', {'in': 0, 'out': 0})]
ref_edges = [('input', 'relu', {'in': 0, 'out': 0}), ('relu', 'rank', {'in': 0, 'out': 0}),
             ('rank', 'range', {'in': 1, 'out': 0}),
             ('relu', 'reduce_mean', {'in': 0, 'out': 0}),
             ('const_1', 'range', {'in': 0, 'out': 0}), ('const_2', 'range', {'in': 2, 'out': 0}),
             ('range', 'reduce_mean', {'in': 1, 'out': 0}),
             ('reduce_mean', 'result', {'in': 0, 'out': 0})]


class GlobalPoolingToReduceTest(unittest.TestCase):
    def test_global_pooling_to_reduce(self):
        graph = build_graph_with_edge_attrs(nodes, edges)

        graph_ref = build_graph(nodes, ref_edges)
        graph.stage = 'front'
        graph.graph['layout'] = 'NCHW'
        node = Node(graph, 'relu')
        node.out_edge(0)['fw_tensor_debug_info'] = [('Relu_0', 'Relu_tensor')]

        GlobalPoolingToReduce().find_and_replace_pattern(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        node = Node(graph, 'relu')
        edge_attrs = node.out_port(0).get_destinations()[0].get_in_edge_attrs()
        self.assertTrue('fw_tensor_debug_info' in edge_attrs)
        self.assertTrue(edge_attrs['fw_tensor_debug_info'] == [('Relu_0', 'Relu_tensor')])
