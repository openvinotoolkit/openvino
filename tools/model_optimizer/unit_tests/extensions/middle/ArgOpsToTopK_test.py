# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.middle.ArgOpsToTopK import ArgOpsToTopK
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import regular_op_with_empty_data, result, build_graph, connect, \
    valued_const_with_data, regular_op, empty_data, connect_front

nodes_attributes = {
    **regular_op_with_empty_data('input', {'op': 'Parameter', 'type': 'Parameter'}),
    **regular_op_with_empty_data('argmax', {'op': 'ArgMax', 'type': None, 'out_max_val': 0, 'top_k': 1, 'axis': 0,
                                            'output_type': np.int32, 'remove_values_output': True}),
    **regular_op_with_empty_data('argmin', {'op': 'ArgMin', 'type': None, 'top_k': 1, 'axis': 0,
                                            'output_type': np.int32, 'remove_values_output': True}),
    **result('result'),
    **valued_const_with_data('axis_const', int64_array([1])),

    **regular_op('topk', {'op': 'TopK', 'type': 'TopK', 'sort': 'index', 'index_element_type': np.int32}),
    **empty_data('topk_out_0_data'),
    **empty_data('topk_out_1_data'),
    **regular_op_with_empty_data('topk_scalar', {'op': 'Const', 'type': 'Const', 'value': int64_array([1]),
                                                 'shape': []}),


    **regular_op_with_empty_data('concat', {'op': 'Concat', 'type': 'Concat', 'axis': 1})
}


class ArgOpsToTopKTest(unittest.TestCase):

    def test_tf_argmax_to_topk(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', '0:argmax'),
                                *connect('axis_const', '1:argmax'),
                                *connect('argmax', 'result')
                            ],
                            nodes_with_edges_only=True)
        ArgOpsToTopK().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:topk'),
                                    *connect('topk_scalar', '1:topk'),
                                    *connect_front('topk:1', 'topk_out_1_data'),
                                    *connect_front('topk_out_1_data', 'result'),
                                ],
                                update_attributes={
                                    'topk': {'axis': int64_array([1]), 'mode': 'max', 'remove_values_output': True},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_tf_argmin_to_topk(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', '0:argmin'),
                                *connect('axis_const', '1:argmin'),
                                *connect('argmin', 'result')
                            ],
                            nodes_with_edges_only=True)
        ArgOpsToTopK().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:topk'),
                                    *connect('topk_scalar', '1:topk'),
                                    *connect_front('topk:1', 'topk_out_1_data'),
                                    *connect_front('topk_out_1_data', 'result')
                                ],
                                update_attributes={
                                    'topk': {'axis': int64_array([1]), 'mode': 'min', 'remove_values_output': True},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_onnx_argmax_to_topk(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', 'argmax'),
                                *connect('argmax', 'result')
                            ],
                            nodes_with_edges_only=True)
        ArgOpsToTopK().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:topk'),
                                    *connect('topk_scalar', '1:topk'),
                                    *connect_front('topk:1', 'topk_out_1_data'),
                                    *connect_front('topk_out_1_data', 'result')
                                ],
                                update_attributes={
                                    'topk': {'axis': 0, 'mode': 'max', 'remove_values_output': True},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_onnx_argmin_to_topk(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', 'argmin'),
                                *connect('argmin', 'result')
                            ],
                            nodes_with_edges_only=True)
        ArgOpsToTopK().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:topk'),
                                    *connect('topk_scalar', '1:topk'),
                                    *connect_front('topk:1', 'topk_out_1_data'),
                                    *connect_front('topk_out_1_data', 'result')
                                ],
                                update_attributes={
                                    'topk': {'axis': 0, 'mode': 'min', 'remove_values_output': True},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_caffe_argmax_to_topk(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', 'argmax'),
                                *connect('argmax', 'result')
                            ],
                            update_attributes={
                                'argmax': {'out_max_val': 1}
                            },
                            nodes_with_edges_only=True)
        ArgOpsToTopK().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:topk'),
                                    *connect('topk_scalar', '1:topk'),
                                    *connect_front('topk:0','topk_out_0_data'),
                                    *connect_front('topk:1', 'topk_out_1_data'),
                                    *connect_front('topk_out_0_data', '1:concat'),
                                    *connect_front('topk_out_1_data', '0:concat'),
                                    *connect('concat', 'result')
                                ],
                                update_attributes={
                                    'topk': {'axis': 0, 'mode': 'max', 'remove_values_output': True},
                                },
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)