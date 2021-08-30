# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.elementwise import Add, Mul
from extensions.ops.mvn import MVN
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class LayerNormalization(FrontReplacementOp):
    """
    Decompose LayerNormalization to scale * (Squeeze - MVN(x) - Unsqueeze) + B
    """
    op = "LayerNorm"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = 1
        if 'axis' in node:
            axis = node.axis
        squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                   dict(name=str(node.id) + '/Squeeze'))
        mvn = MVN(graph, {'eps': node.epsilon, 'name': node.name + '/LayerNorm/MVN_',
                          'across_channels': 1, 'normalize_variance': 1}).create_node()
        unsqueeze = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                     dict(name=str(node.id) + '/Unsqueeze'))
        mul = Mul(graph, {'axis': axis, 'name': node.name + '/LayerNorm/mul_'}).create_node()
        add = Add(graph, {'axis': axis, 'name': node.name + '/LayerNorm/add_'}).create_node()

        node.in_port(0).get_connection().set_destination(squeeze.in_port(0))
        node.in_port(1).get_connection().set_destination(mul.in_port(1))
        node.in_port(2).get_connection().set_destination(add.in_port(1))

        squeeze.out_port(0).get_connection().set_destination(mvn.in_port(0))
        mvn.out_port(0).get_connection().set_destination(unsqueeze.in_port(0))

        unsqueeze.out_port(0).connect(mul.in_port(0))
        mul.out_port(0).connect(add.in_port(0))

        return [add.id]
