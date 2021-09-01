# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.elementwise import Add, Mul
from extensions.ops.mvn import MVN
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class LayerNormalization(MiddleReplacementPattern):
    """
    Decompose LayerNormalization to scale * (Squeeze - MVN(x) - Unsqueeze) + B
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='LayerNorm'):
            axis = 1
            if node.has('axis'):
                axis = node.soft_get('axis')

            input_rank = len(node.in_port(0).data.get_shape())
            if axis < 0:
                axis = input_rank + axis

            name_node = node.soft_get('name', node.id)
            squeeze = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                       dict(name=str(node.id) + '/Squeeze'))
            mvn = create_op_node_with_second_input(graph, MVN, int64_array(axis),
                                             dict(eps=node.epsilon, name=name_node + '/LayerNorm/MVN_',
                                                  across_channels=1, normalize_variance=1, eps_mode='inside_sqrt'))
            unsqueeze = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                         dict(name=str(node.id) + '/Unsqueeze'))
            mul = Mul(graph, {'axis': axis, 'name': name_node + '/LayerNorm/mul_'}).create_node()
            add = Add(graph, {'axis': axis, 'name': name_node + '/LayerNorm/add_'}).create_node()

            node.in_port(0).get_connection().set_destination(squeeze.in_port(0))
            node.in_port(1).get_connection().set_destination(mul.in_port(1))
            node.in_port(2).get_connection().set_destination(add.in_port(1))

            squeeze.out_port(0).get_connection().set_destination(mvn.in_port(0))
            mvn.out_port(0).get_connection().set_destination(unsqueeze.in_port(0))

            unsqueeze.out_port(0).connect(mul.in_port(0))
            mul.out_port(0).connect(add.in_port(0))
            node.out_port(0).get_connection().set_source(add.out_port(0))
