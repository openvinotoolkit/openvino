# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add, Maximum, Minimum, Mul
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Node, Graph


class EltwiseNReplacement(FrontReplacementOp):
    """
    This replacer substitutes elementwise operation with more than 2 inputs with a number of simple elementwise
    operations with 2 inputs. The replacer supports operations supported by the Eltwise layer.
    """
    op = 'EltwiseN'
    enabled = True

    op_to_class_map = {
        'sum': Add,
        'min': Minimum,
        'max': Maximum,
        'mul': Mul,
    }

    def replace_op(self, graph: Graph, node: Node):
        last_node = node
        operation = node.operation
        assert operation in EltwiseNReplacement.op_to_class_map
        op_class = EltwiseNReplacement.op_to_class_map[operation]
        left_connect = node.in_port(0).get_connection()

        for ind in list(node.in_ports())[1:]:
            attrs = {'name': node.name + '/' + operation + '_' + str(ind)}
            attrs.update({'axis': node.axis} if node.has_valid('axis') else {})
            # Create node
            eltwise_op = op_class(graph, attrs).create_node()
            # Connect nodes
            left_connect.set_destination(eltwise_op.in_port(0))
            left_connect = eltwise_op.out_port(0).get_connection()
            node.in_port(ind).get_connection().set_destination(eltwise_op.in_port(1))
            last_node = eltwise_op
        return [last_node.id]
