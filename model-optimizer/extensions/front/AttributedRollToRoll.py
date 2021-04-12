# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.roll import Roll
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const


class AttributedRollToRoll(FrontReplacementPattern):
    """
    This transformation converts AttributedRoll operation (axes and shift are specified as attributes) to Roll
    operation (Inference Engine semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_roll in graph.get_op_nodes(op='AttributedRoll'):
            original_name = attr_roll.soft_get('name', attr_roll.id)

            new_roll = Roll(graph, {'axes': attr_roll.soft_get('axes', None),
                                    'shift': attr_roll.soft_get('shift', None)}).create_node()
            rename_nodes([(attr_roll, original_name + '/to_be_removed'), (new_roll, original_name)])

            attr_roll.in_port(0).get_connection().set_destination(new_roll.in_port(0))
            new_roll.in_port(1).connect(
                Const(graph, {'name': original_name + '/shift', 'value': attr_roll.shift}).create_node().out_port(0))
            if attr_roll.has_valid('axes'):
                new_roll.in_port(2).connect(
                    Const(graph, {'name': original_name + '/axes', 'value': attr_roll.axes}).create_node().out_port(0))

            attr_roll.out_port(0).get_connection().set_source(new_roll.out_port(0))
            graph.remove_node(attr_roll.id)
