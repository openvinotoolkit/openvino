# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Add
from extensions.ops.gather import Gather
from extensions.ops.range import Range
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.ops.shape import Shape
from mo.ops.reshape import Reshape
from mo.ops.squeeze import Squeeze
from mo.utils.error import Error


class ArangeLikeReplacer(FrontReplacementOp):
    op = 'arange_like'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name', node.id)
        if node.repeat != 1:
            raise Error("arange_like node {} with non default repeat value {} "
                        "is not supported".format(name, node.repeat))
        axis = node.soft_get('axis')
        shape_node = Shape(graph, {'name': name + '/shape'}).create_node()
        range_node = create_op_with_const_inputs(graph, Range, {0: int64_array(node.start),
                                                                2: int64_array(1)}, {'name': name + '/Range'})
        shape_node.in_port(0).connect(node.in_port(0).get_source())
        if axis:
            '''
            Replace arange_like op to subgraph:
            Shape - Gather - Range
            '''
            gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis]),
                                                                      2: int64_array(0)},
                                                      {'name': name + '/Gather'})
            shape_node.out_port(0).connect(gather_node.in_port(0))
            gather_node.out_port(0).connect(range_node.in_port(1))
            node.out_port(0).get_connection().set_source(range_node.out_port(0))
            node.in_port(0).disconnect()
        else:
            r'''
            Replace arange_like op to subgraph:
                    |
                  Shape     
                 /     \
            Reshape     |
               |        |
             Range      | 
                \      /
                 Reshape
                    |
            '''

            reshape_node1 = create_op_with_const_inputs(graph, Reshape, {1: int64_array([-1])},
                                                        {'name': name + '/Reshape_forward'})

            reshape_node2 = Reshape(graph, {'name': name + '/Reshape_backward'}).create_node()
            shape_node.out_port(0).connect(reshape_node1.in_port(0))
            reshape_node1.out_port(0).connect(range_node.in_port(1))
            range_node.out_port(0).connect(reshape_node2.in_port(0))
            shape_node.out_port(0).connect(reshape_node2.in_port(1))
            node.out_port(0).get_connection().set_source(reshape_node2.out_port(0))

        # MXNet arange_like op has no stop attribute and the result tensor always matches the input shape, so
        # we have to correct the stop value for the Range node if start > 0
        if node.start > 0:
            correct_stop_value = create_op_with_const_inputs(graph, Add, port_value_dict={1: int64_array(node.start)},
                                                             op_attrs={'name': range_node.name + '/Stop'})
            range_node.in_port(1).get_connection().insert_node(correct_stop_value)

        # Range node supports only scalar inputs
        squeeze_node = create_op_with_const_inputs(graph, Squeeze, port_value_dict={1: int64_array(0)},
                                                   op_attrs={"name": range_node.name + '/Stop/Squeeze'})
        range_node.in_port(1).get_connection().insert_node(squeeze_node)
