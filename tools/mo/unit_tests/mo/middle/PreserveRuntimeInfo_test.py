# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.PreserveRuntimeInfo import PreserveRuntimeInfo
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.ops.transpose import Transpose
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


nodes_for_case_with_two_results = {
    'placeholder1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'placeholder2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add', 'infer': copy_shape_infer},
    'add_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'result1': {'kind': 'op', 'op': 'Result'},
    'result2': {'kind': 'op', 'op': 'Result'},
    'fft': {'kind': 'op', 'op': 'IDFT', 'type': 'IDFT', 'infer': copy_shape_infer},
    'fft_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.float32},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([1]), 'value': int64_array([-1])
    },
    'fft_axes_data': {'value': int64_array([-1]), 'shape': int64_array([1]), 'kind': 'data', 'data_type': np.int64},
    'transpose_parameter_order': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': None, 'value': None
    },
    'transpose_parameter_order_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': np.int64},
    'transpose_parameter': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'infer': Transpose.infer},
    'transpose_parameter_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
}

edges_for_case_with_two_results = [
    ('transpose_parameter_order', 'transpose_parameter_order_data'),
    ('transpose_parameter_order_data', 'transpose_parameter', {'in': 1}),
    ('transpose_parameter', 'transpose_parameter_data'),
    ('placeholder1', 'placeholder1_data'),
    ('placeholder2', 'placeholder2_data'),
    ('placeholder1_data', 'add', {'in': 0}),
    ('placeholder2_data', 'add', {'in': 1}),
    ('add', 'add_data'),
    ('add_data', 'result1', {'out': 0, 'in': 0}),
    ('add_data', 'fft', {'out': 0, 'in': 0}),
    ('fft_axes', 'fft_axes_data'),
    ('fft_axes_data', 'fft', {'in': 1}),
    ('fft', 'fft_data'),
    ('fft_data', 'result2'),
]

edges_with_transpose_for_case_with_two_results = [
    ('transpose_parameter_order', 'transpose_parameter_order_data'),
    ('placeholder1_data', 'transpose_parameter', {'in': 0}),
    ('transpose_parameter_order_data', 'transpose_parameter', {'in': 1}),
    ('transpose_parameter', 'transpose_parameter_data'),
    ('placeholder1', 'placeholder1_data'),
    ('placeholder2', 'placeholder2_data'),
    ('transpose_parameter_data', 'add', {'in': 0}),
    ('placeholder2_data', 'add', {'in': 1}),
    ('add', 'add_data'),
    ('add_data', 'result1', {'out': 0, 'in': 0}),
    ('add_data', 'fft', {'out': 0, 'in': 0}),
    ('fft_axes', 'fft_axes_data'),
    ('fft_axes_data', 'fft', {'in': 1}),
    ('fft', 'fft_data'),
    ('fft_data', 'result2'),
]


class TestPreserveRuntimeInfoTest():
    @pytest.mark.parametrize("nhwc_to_nchw_order, nchw_to_nhwc_order, add_permutation_attrs",[
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
        assert flag, resp

        assert not param_node.has_valid('permute_attrs')
        assert not param_node.out_node(0).has_valid('permutation')

        if add_permutation_attrs:
            rt_info = param_node.rt_info.info
            old_api_map = rt_info[('old_api_map_order', 0)].info
            assert np.array_equal(old_api_map['inverse_order'], nchw_to_nhwc_order)

            rt_info = result_node.rt_info.info
            old_api_map = rt_info[('old_api_map_order', 0)].info
            assert np.array_equal(old_api_map['order'], nhwc_to_nchw_order)

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
        assert flag, resp

        rt_info = param_node.rt_info.info
        old_api_map = rt_info[('old_api_map_order', 0)].info
        assert np.array_equal(old_api_map['inverse_order'], [0, 2, 3, 1])

        rt_info = result_node.rt_info.info
        old_api_map = rt_info[('old_api_map_order', 0)].info
        assert np.array_equal(old_api_map['order'], [0, 3, 1, 2])

    @pytest.mark.parametrize("nhwc_to_nchw_order, nchw_to_nhwc_order,add_permutation_attrs, fft_kind",
                             [([0, 3, 1, 2], [0, 2, 3, 1], True, 'DFT'),
        ([0, 3, 1, 2], [0, 2, 3, 1], True, 'IDFT'),
        (None, None, False, 'DFT'),
        (None, None, False, 'IDFT'),
        ([0, 4, 1, 2, 3], [0, 2, 3, 4, 1], True, 'DFT'),
        ([0, 4, 1, 2, 3], [0, 2, 3, 4, 1], True, 'IDFT'),
    ])
    def test_transpose_insert_with_two_result_nodes(self, nhwc_to_nchw_order, nchw_to_nhwc_order,
                                                    add_permutation_attrs, fft_kind):
        shape_len = len(nhwc_to_nchw_order) if add_permutation_attrs else 3
        shape = np.array(range(shape_len))
        add_shape = shape if nhwc_to_nchw_order is None else shape[nhwc_to_nchw_order]
        graph = build_graph(nodes_attrs=nodes_for_case_with_two_results,
                            edges=edges_for_case_with_two_results,
                            update_attributes={
                                'placeholder1_data': {'shape': int64_array(shape)},
                                'placeholder1': {'shape': int64_array(shape), 'rt_info': RTInfo()},
                                'transpose_parameter_order': {
                                    'value': np.array(nhwc_to_nchw_order),
                                    'shape': int64_array(np.array(nhwc_to_nchw_order).shape)
                                },
                                'transpose_parameter_order_data': {
                                    'value': np.array(nhwc_to_nchw_order),
                                    'shape': int64_array(np.array(nhwc_to_nchw_order).shape)
                                },
                                'fft': {'op': fft_kind, 'type': fft_kind},
                                'add_data': {'shape': add_shape},
                                'fft_data': {'shape': add_shape},
                                'result1': {'shape': shape, 'rt_info': RTInfo()},
                                'result2': {'shape': shape, 'rt_info': RTInfo()},
                            })

        if add_permutation_attrs:
            graph_ref = build_graph(nodes_for_case_with_two_results, edges_with_transpose_for_case_with_two_results)
        else:
            graph_ref = build_graph(nodes_for_case_with_two_results, edges_for_case_with_two_results)

        param1_node = Node(graph, 'placeholder1')
        result1_node = Node(graph, 'result1')
        result2_node = Node(graph, 'result2')

        if add_permutation_attrs:
            shape_len = len(nhwc_to_nchw_order)
            param1_node['permute_attrs'] = PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])
            param1_node.out_node(0)['permutation'] = PermuteAttrs().get_nhwc_to_nchw_permutation(shape_len)
            result1_node.in_node(0)['permutation'] = PermuteAttrs().get_nhwc_to_nchw_permutation(shape_len)
            result2_node.in_node(0)['permutation'] = PermuteAttrs().get_nhwc_to_nchw_permutation(shape_len)

        PreserveRuntimeInfo().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result1')
        assert flag, resp

        assert not param1_node.has_valid('permute_attrs')
        assert not param1_node.out_node(0).has_valid('permutation')

        if add_permutation_attrs:
            rt_info = param1_node.rt_info.info
            old_api_map = rt_info[('old_api_map_order', 0)].info
            assert np.array_equal(old_api_map['inverse_order'], nchw_to_nhwc_order)
