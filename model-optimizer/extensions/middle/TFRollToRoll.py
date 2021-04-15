# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.roll import Roll
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern


def convert_roll(graph: Graph, roll: Node):
    roll_name = roll.soft_get('name', roll.id)
    new_roll = Roll(graph, {}).create_node()

    roll.in_port(0).get_connection().set_destination(new_roll.in_port(0))
    roll.in_port(1).get_connection().set_destination(new_roll.in_port(1))
    roll.in_port(2).get_connection().set_destination(new_roll.in_port(2))

    roll.out_port(0).get_connection().set_source(new_roll.out_port(0))

    rename_nodes([(roll, roll_name + '/to_be_removed'), (new_roll, roll_name)])


class TFRollToRoll(MiddleReplacementPattern):
    """
    The transformation replaces TFRoll with Roll.
    """
    enabled = True

    def run_after(self):
        from extensions.middle.CorrectRollAxes import CorrectRollAxes
        return [CorrectRollAxes]

    def find_and_replace_pattern(self, graph: Graph):
        roll_ops = graph.get_op_nodes(op='TFRoll')
        for roll in roll_ops:
            convert_roll(graph, roll)
