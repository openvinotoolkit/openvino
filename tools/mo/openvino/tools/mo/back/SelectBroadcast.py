# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.ReshapeMutation import ReshapeMutation
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class SelectBroadcast(BackReplacementPattern):
    """
    Select broadcasting semantics in TF isn't numpy-like
    broadcasting rules, manual reshape is needed.
    For example:
        condition: [1]
        input_1: [1, 8]
        input_2: [1, 8]
    Condition should be aligned with first dimensions of inputs.
    """
    enabled = True

    def run_before(self):
        return [ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='Select'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        select = match['op']
        if select.has_valid('format') and select['format'] == 'tf':
            condition = select.in_node(0)
            input_1 = select.in_node(1)

            if len(condition.shape) == 1 and len(input_1.shape) > 1:
                unsqueeze_op = create_op_node_with_second_input(
                    graph, Unsqueeze, int64_array(range(1, len(input_1.shape))),
                    {'name': select.name+'/Broadcast/'}, select.in_port(0).get_source().node)

                select.in_port(0).disconnect()
                select.in_port(0).get_connection().set_source(unsqueeze_op.out_port(0))
