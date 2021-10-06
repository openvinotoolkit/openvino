# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.AttributedRollToRoll import AttributedRollToRoll
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape


class RollWithEmptyAxesReplacer(FrontReplacementPattern):
    """
    According to MxNet Roll specification axes is an optional parameter. If it is not specified
    input tensor is flattened, then Roll is applied along 0 axis and then the resulting tensor
    is reshaped to original shape.
    This transformation replaces Roll with empty axes input with the following sequence of operations:
    reshape to 1D tensor -> Roll -> reshape to original shape.
    """
    enabled = True

    def run_after(self):
        return [AttributedRollToRoll]

    def find_and_replace_pattern(self, graph: Graph):
        for roll_node in graph.get_op_nodes(op='Roll'):
            if not roll_node.in_port(2).disconnected():
                return
            node_name = roll_node.soft_get('name', roll_node.id)

            # reshape to 1d tensor
            reshape_to_1d = create_op_node_with_second_input(graph, Reshape, int64_array([-1]),
                                                             {'name': node_name + '/reshape'})
            roll_node.in_port(0).get_connection().insert_node(reshape_to_1d)

            # add zero const as axes input to roll
            const_zero = Const(graph, {'value': int64_array([0]), 'name': node_name + '/axes'}).create_node()
            const_zero.out_port(0).connect(roll_node.in_port(2))

            # reshape to original shape
            shape_of = Shape(graph, {'name': node_name + '/shape_of'}).create_node()
            reshape_to_1d.in_port(0).get_connection().add_destination(shape_of.in_port(0))
            reshape_to_orig_shape = Reshape(graph, {}).create_node()
            rename_nodes([(roll_node, node_name + '/roll'), (reshape_to_orig_shape, node_name)])
            shape_of.out_port(0).connect(reshape_to_orig_shape.in_port(1))
            roll_node.out_port(0).get_connection().insert_node(reshape_to_orig_shape)
