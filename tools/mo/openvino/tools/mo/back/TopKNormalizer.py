# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.result import Result


class TopKNormalizer(BackReplacementPattern):
    """
    The transformation converts the second input to the TopK layer from 0D to 1D.

    Also the transformation adds the Result Op if there are no consumers of TopK outputs. However the Result for output
    with values is not added if the node has attribute 'remove_values_output' which is set to True for Caffe models
    where ArgMax does not have separate output with values.

    TODO this pass should be removed when OV supports 0D tensors.
    """
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('result', {'type': 'TopK'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['result']

        reshape = create_op_node_with_second_input(graph, Reshape, int64_array([]), {'override_output_shape': True})
        node.in_port(1).get_connection().insert_node(reshape)

        TopKNormalizer.normalize_outputs(node)

    @staticmethod
    def normalize_outputs(node: Node):
        """
        This function adds missed outputs for TopK node.
        """
        if node.out_port(0).disconnected():
            output = Result(node.graph, {'name': node.name + '/Result_port_0/',
                                         'keep_output_port': node.has_and_set('remove_values_output')}).create_node()
            node.out_port(0).get_connection().set_destination(output.in_port(0))
        if node.out_port(1).disconnected():
            output = Result(node.graph, {'name': node.name + '/Result_port_1/'}).create_node()
            node.out_port(1).get_connection().set_destination(output.in_port(0))
