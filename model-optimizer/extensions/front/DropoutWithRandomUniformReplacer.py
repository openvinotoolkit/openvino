# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.pattern_match import check_value
from mo.ops.broadcast import Broadcast


class DropoutWithRandomUniformReplacer(FrontReplacementSubgraph):
    r"""
    This transformation replaces possible Dropout block (in inference mode) with RandomUniform
    to Broadcast of half-ones in a sub-graph.
    WARNING: the transformation can be triggered for other block with RandomUniform by mistake,
    i.e. replace the detected sub-graph to functionally non-equivalent sub-graph

    Dropout block:
    ShapeOf -> RandomUniform -> Mul ---> Add ---> Add -> Floor
                                         /\
                                         |
                                     Const(0)

    Resulted block:
    ShapeOf --> Broadcast --> Mul ---> Add ---> Add -> Floor
                  /\                   /\
                  |                    |
               Const(0.5)           Const(0)
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('shape', dict(op='ShapeOf')),
                ('random_uniform', dict(op='RandomUniform')),
                ('mul', dict(op='Mul')),
                ('add_const', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 0.0, atol=0)))),
                ('add', dict(op='Add')),
                ('add2', dict(op='Add')),
                ('floor', dict(op='Floor')),
            ],
            edges=[
                ('shape', 'random_uniform'),
                ('random_uniform', 'mul'),
                ('mul', 'add', {'in': 0}),
                ('add_const', 'add', {'in': 1}),
                ('add', 'add2'),
                ('add2', 'floor'),
            ]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict, **kwargs):
        random_uniform_node = match['random_uniform']
        random_uniform_node_name = random_uniform_node.soft_get('name', random_uniform_node.id)
        log.error("Possible dropout block with RandomUniform is detected. "
                  "Replace {} with a Broadcast with constant value of 0.5 "
                  "assuming that it is executed in inference mode.".format(random_uniform_node_name),
                  extra={'is_warning': True})
        data_type = match['add_const'].data_type
        broadcast_node = create_op_with_const_inputs(graph, Broadcast,
                                                     {0: np.array([0.5], dtype=data_type)},
                                                     {'mode': 'numpy',
                                                      'name': random_uniform_node_name + '/Broadcast'})
        rename_nodes([(random_uniform_node, random_uniform_node_name + '/ToBeRemoved'),
                      (broadcast_node, random_uniform_node_name)])
        random_uniform_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
        random_uniform_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
