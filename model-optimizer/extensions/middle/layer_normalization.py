# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.elementwise import Add, Mul
from extensions.ops.mvn import MVN
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class LayerNormalization(MiddleReplacementPattern):
    """
    Decompose LayerNorm(x) to MVN(x) * gamma + beta
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='LayerNorm'):
            node_name = node.soft_get('name', node.id)
            assert node.has_valid('axis'), 'Incorrect axis value {} for the node {}'.format(node.axis, node_name)
            axis = node.axis

            mvn = create_op_node_with_second_input(graph, MVN, int64_array([axis]),
                                                   dict(eps=node.epsilon, name=node_name + '/LayerNorm/MVN_',
                                                        across_channels=1, normalize_variance=1, eps_mode='inside_sqrt'))

            mul = Mul(graph, {'name': node_name + '/LayerNorm/mul_'}).create_node()
            add = Add(graph, {'name': '/LayerNorm/add_'}).create_node()

            node.in_port(0).get_connection().set_destination(mvn.in_port(0))
            node.in_port(1).get_connection().set_destination(mul.in_port(1))
            node.in_port(2).get_connection().set_destination(add.in_port(1))

            mvn.out_port(0).connect(mul.in_port(0))
            mul.out_port(0).connect(add.in_port(0))
            node.out_port(0).get_connection().set_source(add.out_port(0))

            rename_nodes([(node, node_name + '/ShouldBeDeleted'), (add, node_name)])
