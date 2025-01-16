# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node


class RemoveLastSoftMaxPattern(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi' and graph.graph['cmd_params'].remove_output_softmax]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax_node', dict(op='SoftMax')),
                ('softmax_data', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('softmax_node', 'softmax_data'),
                ('softmax_data', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Removes output SoftMax layer
        :param graph: graph to operate on
        :param match: dictionary with matched nodes
        """
        if len(match['softmax_data'].out_nodes()) == 1:
            remove_op_node_with_data_node(graph, match['softmax_node'])
        else:
            log.error("SoftMax is not last layer, so can't be removed", extra={'is_warning': True})


class RemoveLastLogSoftMaxPattern(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi' and graph.graph['cmd_params'].remove_output_softmax]
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('input_data', {'kind': 'data'}),
                ('sub_node', {'kind': 'op', 'op': 'Sub'}),
                ('reduce_max_node', {'kind': 'op', 'op': 'ReduceMax'}),
                ('reduce_max_node_data', {'kind': 'data'}),
                ('sub_node_data', {'kind': 'data'}),
                ('exp', {'kind': 'op', 'op': 'Exp'}),
                ('exp_data', {'kind': 'data'}),
                ('reduce_sum_node', {'kind': 'op', 'op': 'ReduceSum'}),
                ('reduce_sum_node_data', {'kind': 'data'}),
                ('reduce_sum_axis', {'kind': 'op', 'op': 'Const'}),
                ('reduce_sum_axis_data', {'kind': 'data'}),
                ('log', {'kind': 'op', 'op': 'Log'}),
                ('log_data', {'kind': 'data'}),
                ('last_sub', {'kind': 'op', 'op': 'Sub'}),
                ('last_sub_data', {'kind': 'data'}),
                ('op_output', {'kind': 'op', 'op': 'Result'}),
            ],
            edges=[
                ('input_data', 'sub_node', {'in': 0}),
                ('input_data', 'reduce_max_node', {'in': 0}),
                ('reduce_max_node', 'reduce_max_node_data'),
                ('reduce_max_node_data', 'sub_node', {'in': 1}),
                ('sub_node', 'sub_node_data'),
                ('sub_node_data', 'exp', {'out': 0, 'in': 0}),
                ('exp', 'exp_data'),
                ('exp_data', 'reduce_sum_node', {'in': 0}),
                ('reduce_sum_node', 'reduce_sum_node_data'),
                ('reduce_sum_axis', 'reduce_sum_axis_data'),
                ('reduce_sum_axis_data', 'reduce_sum_node', {'in': 1}),
                ('reduce_sum_node_data', 'log'),
                ('log', 'log_data'),
                ('log_data', 'last_sub', {'in': 1}),
                ('last_sub', 'last_sub_data'),
                ('sub_node_data', 'last_sub', {'out': 0, 'in': 0}),
                ('last_sub_data', 'op_output'),
            ]
        )

    expected_number_of_outputs = {
        'reduce_max_node': 1, 'reduce_sum_node': 1, 'exp': 1, 'log': 1, 'sub_node': 2, 'last_sub': 1
    }

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Removes output LogSoftMax layer
        :param graph: graph to operate on
        :param match: dictionary with matched nodes
        """
        reduce_max_node = match['reduce_max_node']
        second_input_of_reduce_max = reduce_max_node.in_port(1).get_connection().get_source().node
        if not second_input_of_reduce_max.has_valid('value') or len(second_input_of_reduce_max.value) != 1:
            return

        reduce_sum_node = match['reduce_sum_node']
        second_input_of_reduce_sum = reduce_sum_node.in_port(1).get_connection().get_source().node
        if not second_input_of_reduce_sum.has_valid('value') or len(second_input_of_reduce_sum.value) != 1:
            return
        if second_input_of_reduce_max.value[0] != second_input_of_reduce_sum.value[0]:
            return

        for name, number in RemoveLastLogSoftMaxPattern.expected_number_of_outputs.items():
            if len(match[name].out_port(0).get_destinations()) != number:
                return

        match['op_output'].in_port(0).get_connection().set_source(match['sub_node'].in_port(0).get_source())
