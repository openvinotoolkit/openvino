# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.const import Const


class ConstantOfShapeToBroadcast(FrontReplacementPattern):
    """
    Converts the 'ConstantOfShape' layer to 'Broadcast'.

    The 'ConstantOfShape' has one 1D input defining the output constant shape. The value to be filled is defined by the
    'value' attribute. The transformation creates constant node with value equal to 'value' attribute and connects it to
    the first input of a newly created 'Broadcast' node which defines value to broadcast. Then the input of the
    'ConstantOfShape' is connected to the second input of the 'Broadcast'.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for const_of_shape_node in graph.get_op_nodes(op='ConstantOfShape'):
            broadcast_node = Broadcast(graph, {'name': const_of_shape_node.name + '/Broadcast'}).create_node()
            const_of_shape_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
            broadcast_node.in_port(0).connect(Const(graph, {'name': broadcast_node.name + '/FillValue',
                                                            'value': const_of_shape_node.fill_value}
                                                    ).create_node().out_port(0))
            const_of_shape_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
