# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.middle.PreserveRuntimeInfo import PreserveRuntimeInfo
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.utils.runtime_info import RTInfo
from unit_tests.utils.graph import build_graph, connect, valued_const_with_data, regular_op_with_empty_data, \
    regular_op_with_shaped_data

nodes = {
    **regular_op_with_empty_data('placeholder2', {'type': 'Parameter'}),
    **regular_op_with_empty_data('transpose_parameter',
                                 {'type': 'Transpose', 'op': 'Transpose', 'infer': Transpose.infer}),
    **regular_op_with_empty_data('transpose_result',
                                 {'type': 'Transpose', 'op': 'Transpose', 'infer': Transpose.infer}),
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
        shape_len = len(nhwc_to_nchw_order) if add_permutation_attrs else 3
        shape = np.array(range(shape_len))
        add_shape = shape if nhwc_to_nchw_order is None else shape[nhwc_to_nchw_order]
        graph_nodes.update(
            {
                **regular_op_with_shaped_data('placeholder1', shape,
                                              {'type': 'Parameter', 'rt_info': RTInfo(), 'shape': shape}),
                **regular_op_with_shaped_data('result', shape, {'type': 'Result', 'rt_info': RTInfo(), 'shape': shape}),
                **regular_op_with_shaped_data('add', add_shape,
                                              {'type': 'Add', 'op': 'Add', 'infer': copy_shape_infer}),
            }
        )

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

        if add_permutation_attrs:
            rt_info = param_node.rt_info.info
            old_api_map = rt_info[('old_api_map_order', 0)].info
            self.assertTrue(np.array_equal(old_api_map['inverse_order'], nchw_to_nhwc_order))

            rt_info = result_node.rt_info.info
            old_api_map = rt_info[('old_api_map_order', 0)].info
            self.assertTrue(np.array_equal(old_api_map['order'], nhwc_to_nchw_order))

    def test_auto_disable_nhwc_to_nchw(self):
        shape_len = 4
        shape = np.array(range(shape_len))
        add_shape = shape
        graph_nodes = {
            **regular_op_with_shaped_data('placeholder1', shape,
                                          {'type': 'Parameter', 'rt_info': RTInfo(), 'shape': shape}),
            **regular_op_with_shaped_data('placeholder2', shape,
                                          {'type': 'Parameter', 'rt_info': RTInfo(), 'shape': shape}),
            **regular_op_with_shaped_data('result', shape, {'type': 'Result', 'rt_info': RTInfo(), 'shape': shape}),
            **regular_op_with_shaped_data('add', add_shape,
                                          {'type': 'Add', 'op': 'Add', 'infer': copy_shape_infer}),
        }

        graph = build_graph(graph_nodes, edges)
        graph.graph['cmd_params'].auto_disable_nhwc_to_nchw = True
        graph_ref = build_graph(graph_nodes, edges)

        param_node = Node(graph, 'placeholder1')
        result_node = Node(graph, 'result')

        PreserveRuntimeInfo().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

        rt_info = param_node.rt_info.info
        old_api_map = rt_info[('old_api_map_order', 0)].info
        self.assertTrue(np.array_equal(old_api_map['inverse_order'], [0, 2, 3, 1]))

        rt_info = result_node.rt_info.info
        old_api_map = rt_info[('old_api_map_order', 0)].info
        self.assertTrue(np.array_equal(old_api_map['order'], [0, 3, 1, 2]))
