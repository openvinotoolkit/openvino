# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.AttributedRollToRoll import AttributedRollToRoll
from extensions.ops.roll import Roll
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape


class RollWithEmptyAxesReplacer(FrontReplacementPattern):
    """
    This transformation replaces Roll with empty axes input with the following sequence of operations:
    reshape to 1D tensor -> roll -> reshape to original shape.
    """
    enabled = True

    def run_after(self):
        return [AttributedRollToRoll]

    def find_and_replace_pattern(self, graph: Graph):
        for roll_node in graph.get_op_nodes(op='Roll'):
            if len(roll_node.in_ports()) < 3 or not roll_node.in_port(2).disconnected():
                continue
            node_name = roll_node.soft_get('name', roll_node.id)

            # reshape to 1d tensor
            reshape_to_1d = create_op_node_with_second_input(graph, Reshape, int64_array([-1]),
                                                             {'name': node_name + '/reshape'})
            roll_node.in_port(0).get_connection().set_destination(reshape_to_1d.in_port(0))

            # roll
            const_zero = Const(graph, {'value': int64_array([0]), 'name': node_name + '/axes'}).create_node()
            new_roll = Roll(graph, {'name': node_name + '/roll'}).create_node()
            reshape_to_1d.out_port(0).connect(new_roll.in_port(0))
            roll_node.in_port(1).get_connection().set_destination(new_roll.in_port(1))
            const_zero.out_port(0).connect(new_roll.in_port(2))

            # reshape to original shape
            shape_of = Shape(graph, {'name': node_name + '/shape_of'}).create_node()
            reshape_to_orig_shape = Reshape(graph, {}).create_node()
            rename_nodes([(roll_node, node_name + '/to_be_removed'), (reshape_to_orig_shape, node_name)])
            reshape_to_1d.in_port(0).get_connection().add_destination(shape_of.in_port(0))
            new_roll.out_port(0).connect(reshape_to_orig_shape.in_port(0))
            shape_of.out_port(0).connect(reshape_to_orig_shape.in_port(1))
            roll_node.out_port(0).get_connection().set_source(reshape_to_orig_shape.out_port(0))

            graph.remove_node(roll_node.id)
