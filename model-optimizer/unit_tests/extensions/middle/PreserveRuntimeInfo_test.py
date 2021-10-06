# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from extensions.middle.PreserveRuntimeInfo import PreserveRuntimeInfo
from mo.graph.graph import Node
from mo.ops.op import PermuteAttrs
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.runtime_info import RTInfo
from unit_tests.utils.graph import build_graph, connect, valued_const_with_data, regular_op_with_empty_data

nodes = {
    **regular_op_with_empty_data('placeholder1', {'type': 'Parameter', 'rt_info': RTInfo()}),
    **regular_op_with_empty_data('placeholder2', {'type': 'Parameter'}),
    **regular_op_with_empty_data('add', {'type': 'Add', 'op': 'Add'}),
    **regular_op_with_empty_data('result', {'type': 'Result', 'rt_info': RTInfo()}),

    **regular_op_with_empty_data('transpose_parameter', {'type': 'Transpose', 'op': 'Transpose'}),
    **regular_op_with_empty_data('transpose_result', {'type': 'Transpose', 'op': 'Transpose'}),
}

edges = [*connect('placeholder1', '0:add'), *connect('placeholder2', '1:add'), *connect('add', 'result')]
edges_with_transpose = [*connect('placeholder1', '0:transpose_parameter'),
                        *connect('transpose_parameter_order', '1:transpose_parameter'),
                        *connect('transpose_parameter', '0:add'),
                        *connect('placeholder2', '1:add'),
                        *connect('add', '0:transpose_result'),
                        *connect('transpose_result_order', '1:transpose_result'),
                        *connect('transpose_result', 'result')]


@generator
class PreserveRuntimeInfoTest(unittest.TestCase):
    @generate(*[
        ([0, 3, 1, 2], [0, 2, 3, 1], True),
        ([0, 4, 1, 2, 3], [0, 2, 3, 4, 1], True),
        (None, None, False),
    ])
    def test_transpose_insert(self, nhwc_to_nchw_order, nchw_to_nhwc_order, add_permutation_attrs):
        graph_nodes = {
            **valued_const_with_data('transpose_parameter_order', np.array(nhwc_to_nchw_order)),
            **valued_const_with_data('transpose_result_order', np.array(nchw_to_nhwc_order))
        }
        graph_nodes.update(nodes)
        graph = build_graph(graph_nodes, edges)
        graph_ref = build_graph(graph_nodes, edges_with_transpose if add_permutation_attrs else edges)

        param_node = Node(graph, 'placeholder1')
        result_node = Node(graph, 'result')

        if add_permutation_attrs:
            shape_len = len(nhwc_to_nchw_order)
            param_node['permute_attrs'] = PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])
            param_node.out_node(0)['permutation'] = PermuteAttrs().get_nhwc_to_nchw_permutation(shape_len)
            result_node.in_node(0)['permutation'] = PermuteAttrs().get_nhwc_to_nchw_permutation(shape_len)

        PreserveRuntimeInfo().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        self.assertFalse(param_node.has_valid('permute_attrs'))
        self.assertFalse(param_node.out_node(0).has_valid('permutation'))
        self.assertFalse(result_node.in_node(0).has_valid('permutation'))

        if add_permutation_attrs:
            rt_info = param_node.rt_info.info
            old_api_map = rt_info[('old_api_map', 0)].info
            self.assertTrue(np.array_equal(old_api_map['inverse_order'], nchw_to_nhwc_order))

            rt_info = result_node.rt_info.info
            old_api_map = rt_info[('old_api_map', 0)].info
            self.assertTrue(np.array_equal(old_api_map['order'], nhwc_to_nchw_order))
