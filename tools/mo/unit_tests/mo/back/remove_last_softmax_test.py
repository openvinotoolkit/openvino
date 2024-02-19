# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.remove_last_softmax_pattern import RemoveLastSoftMaxPattern, RemoveLastLogSoftMaxPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class KaldiRemoveLastSoftMaxTest(unittest.TestCase):
    nodes = {
        'input_node': {
            'kind': 'data'
        },
        'softmax_node': {
            'op': 'SoftMax',
            'kind': 'op'
        },
        'output_node': {
            'kind': 'data'
        },
        'op_output': {
            'kind': 'op',
            'op': 'Result'
        },
        'log_node': {
            'op': 'Log',
            'kind': 'op'
        },
        'log_data': {
            'kind': 'data'
        },
    }

    nodes_for_logsoftmax = {
        'input': {'kind': 'op', 'op': 'Parameter'},
        'input_data': {'kind': 'data'},
        'sub': {'kind': 'op', 'op': 'Sub'},
        'reduce_max_node': {'kind': 'op', 'op': 'ReduceMax'},
        'reduce_max_node_data': {'kind': 'data'},
        'reduce_max_axis': {
            'kind': 'op',
            'op': 'Const',
            'type': 'Const',
            'value': int64_array([1]),
            'shape': int64_array([1]),
        },
        'reduce_max_axis_data': {
            'kind': 'data',
            'value': int64_array([1]),
            'shape': int64_array([1]),
        },
        'sub_data': {'kind': 'data'},
        'exp': {'kind': 'op', 'op': 'Exp'},
        'exp_data': {'kind': 'data'},
        'reduce_sum_node': {'kind': 'op', 'op': 'ReduceSum'},
        'reduce_sum_node_data': {'kind': 'data'},
        'reduce_sum_axis': {
            'kind': 'op',
            'op': 'Const',
            'type': 'Const',
            'value': int64_array([1]),
            'shape': int64_array([1]),
        },
        'reduce_sum_axis_data': {
            'kind': 'data',
            'value': int64_array([1]),
            'shape': int64_array([1]),
        },
        'log': {'kind': 'op', 'op': 'Log'},
        'log_data': {'kind': 'data'},
        'last_sub': {'kind': 'op', 'op': 'Sub'},
        'last_sub_data': {'kind': 'data'},
        'op_output': {'kind': 'op', 'op': 'Result'},
    }

    edges_for_logsoftmax = [
        ('input', 'input_data'),
        ('input_data', 'sub', {'in': 0}),
        ('input_data', 'reduce_max_node', {'in': 0}),
        ('reduce_max_node', 'reduce_max_node_data'),
        ('reduce_max_node_data', 'sub', {'in': 1}),
        ('reduce_max_axis', 'reduce_max_axis_data'),
        ('reduce_max_axis_data', 'reduce_max_node', {'in': 1}),
        ('sub', 'sub_data'),
        ('sub_data', 'exp', {'out': 0, 'in': 0}),
        ('exp', 'exp_data'),
        ('exp_data', 'reduce_sum_node', {'in': 0}),
        ('reduce_sum_node', 'reduce_sum_node_data'),
        ('reduce_sum_axis', 'reduce_sum_axis_data'),
        ('reduce_sum_axis_data', 'reduce_sum_node', {'in': 1}),
        ('reduce_sum_node_data', 'log'),
        ('log', 'log_data'),
        ('log_data', 'last_sub', {'in': 1}),
        ('last_sub', 'last_sub_data'),
        ('sub_data', 'last_sub', {'out': 0, 'in': 0}),
        ('last_sub_data', 'op_output'),
    ]

    def test_remove_last_SoftMax(self):
        graph = build_graph(self.nodes, [
            ('input_node', 'softmax_node'),
            ('softmax_node', 'output_node'),
            ('output_node', 'op_output')
        ], nodes_with_edges_only=True)
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        self.assertNotIn('softmax_node', graph.node)

    def test_remove_last_LogSoftMax(self):
        graph = build_graph(nodes_attrs=self.nodes_for_logsoftmax, edges=self.edges_for_logsoftmax)
        RemoveLastLogSoftMaxPattern().find_and_replace_pattern(graph)
        graph.clean_up()

        ref_graph_nodes_attributes = {
            'input': {'kind': 'op', 'op': 'Parameter'},
            'input_data': {'kind': 'data'},
            'op_output': {'kind': 'op', 'op': 'Result'},
        }

        ref_graph_edges = [('input', 'input_data'), ('input_data', 'op_output')]
        ref_graph = build_graph(ref_graph_nodes_attributes, ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'op_output')
        self.assertTrue(flag, resp)

    def test_do_not_remove_not_last_SoftMax(self):
        graph = build_graph(self.nodes, [
            ('input_node', 'softmax_node'),
            ('softmax_node', 'output_node')
        ])
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        self.assertIn('softmax_node', graph.node)
