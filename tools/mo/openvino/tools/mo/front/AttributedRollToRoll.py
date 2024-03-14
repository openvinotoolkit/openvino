# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.roll import Roll
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class AttributedRollToRoll(FrontReplacementPattern):
    """
    This transformation converts AttributedRoll operation (axes and shift are specified as attributes) to Roll
    operation (OpenVINO semantic).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_roll in graph.get_op_nodes(op='AttributedRoll'):
            original_name = attr_roll.soft_get('name', attr_roll.id)
            port_value_dict = {1: attr_roll.shift}
            if attr_roll.has_valid('axes'):
                port_value_dict.update({2: attr_roll.axes})

            new_roll = create_op_with_const_inputs(graph, op=Roll, port_value_dict=port_value_dict)
            rename_nodes([(attr_roll, original_name + '/to_be_removed'), (new_roll, original_name)])

            attr_roll.in_port(0).get_connection().set_destination(new_roll.in_port(0))
            attr_roll.out_port(0).get_connection().set_source(new_roll.out_port(0))
            graph.remove_node(attr_roll.id)
