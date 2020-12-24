"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.broadcast import Broadcast



class DropoutWithRandomUniformReplacer(FrontReplacementSubgraph):
    """
    This transformation replaces possible Dropout block (in inference mode) with RandomUniform
    with Broadcast of ones in a sub-graph.
    WARNING: the transformation can be triggered for other block with RandomUniform by mistake,
    i.e. replace the detected sub-graph to functionally non-equivalent sub-graph

    Dropout block:
    ShapeOf -> RandomUniform -> Mul ---> Add ---> Add -> Floor
                                         /\
                                         |
                                     Const(0)

    Resulted block:
    ShapeOf -> Broadcast
                   /\
                   |
               Const(1)
    """
    enabled = True

    def run_before(self):
        return []

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('shape', dict(op='ShapeOf')),
                ('random_uniform', dict(op='RandomUniform')),
                ('mul', dict(op='Mul')),
                ('add_const', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 0.0, atol=0))),
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
        log.warning('Possible dropout block with RandomUniform is detected.')
        random_uniform_node = match['random_uniform']
        floor_node = match['floor']
        floor_node_name = floor_node.soft_get('name', floor_node.id)
        data_type = match['add_const'].data_type
        broadcast_node = create_op_with_const_inputs(graph, Broadcast,
                                                     {0: np.array([1], dtype=data_type)},
                                                     {'mode': 'numpy', 'name': floor_node_name + '/Broadcast'})
        rename_nodes([(floor_node, floor_node_name + '/ToBeRemoved'), (broadcast_node, floor_node_name)])
        random_uniform_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
        floor_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
        pass
