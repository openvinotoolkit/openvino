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
import numpy as np

from typing import Dict

from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.broadcast import Broadcast


class RandomUniformReplacer(MiddleReplacementPattern):
    """
    Replaces RandomUniform operation with Broadcast of ones in sub-graph:

    ShapeOf ---> RandomUniform ---> Mul

    """

    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('shape', dict(kind='op', op='ShapeOf')),
                ('shape_data', dict(kind='data')),
                ('random_uniform', dict(kind='op', op='RandomUniform')),
                ('random_uniform_data', dict(kind='data')),
                ('mul', dict(kind='op', op='Mul'))
            ],
            edges=[
                ('shape', 'shape_data'),
                ('shape_data', 'random_uniform'),
                ('random_uniform', 'random_uniform_data'),
                ('random_uniform_data', 'mul', {'in': 0})
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: Dict[str, Node]):
        node = match['random_uniform']
        node_name = node.soft_get('name', node.id)
        data_type = node.in_port(0).get_data_type()
        broadcast_node = create_op_with_const_inputs(graph, Broadcast, port_value_dict={0: np.array([1], dtype=data_type)},
                                                     op_attrs={'name': 'Broadcast'})
        node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
        node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))

        rename_nodes([(node, node_name + '/ShouldBeDeleted'), (broadcast_node, node_name)])
