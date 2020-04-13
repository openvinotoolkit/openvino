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

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Div
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node, rename_node
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_features_dimension_value, node_to_get_batch_value, \
    new_shape_node_from_shape_nodes


class ShuffleChannel(FrontReplacementPattern):
    """
    Before:
        ShuffleChannel(group)

    After:
        Reshape[input_batch, group, input_channels/group, -1]
          \/
        Transpose[0, 2, 1, 3]
          \/
        Reshape[input_shape]
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    @staticmethod
    def decompose_shuffle_channel(node: Node):
        graph = node.graph
        name = node.soft_get('name', node.id)

        rename_node(node, name + '/to_be_removed')

        shape = Shape(graph, dict(name=name + '/InputShape')).create_node()
        shape.in_port(0).connect(node.in_port(0).get_source())

        # Reshape [input_batch, group, input_channels/group, -1]
        batch = node_to_get_batch_value(shape)
        group = Const(graph, dict(name=name + '/Rows', value=int64_array([node.group]))).create_node()
        const = Const(graph, dict(name=name + '/Const', value=int64_array([-1]))).create_node()

        input_channels = node_to_get_features_dimension_value(shape)
        output_channels = create_op_node_with_second_input(
            graph, Div, np.int64(node.group), {'name': name + '/Cols'}, input_node=input_channels)
        i_output_channels = Cast(graph, {'name': output_channels.name + '/Convert', 'dst_type': np.int64}).create_node()
        output_channels.out_port(0).connect(i_output_channels.in_port(0))

        reshape_split_dim = new_shape_node_from_shape_nodes([batch, group, i_output_channels, const])
        reshape_split_node = Reshape(graph, dict(name=name + '/Reshape_split_')).create_node()
        reshape_split_dim.out_port(0).connect(reshape_split_node.in_port(1))

        # Transpose(0, 2, 1, 3)
        transpose_node = create_op_node_with_second_input(
            graph, Transpose, int64_array([0, 2, 1, 3]), {'name': name + '/Transpose_'}, input_node=reshape_split_node)

        # Reshape back to input shape
        reshape_concat = Reshape(graph, dict(name=name)).create_node()
        rename_node(reshape_concat, name)

        shape.out_port(0).connect(reshape_concat.in_port(1))
        transpose_node.out_port(0).connect(reshape_concat.in_port(0))

        # Final connections
        node.in_port(0).get_connection().set_destination(reshape_split_node.in_port(0))
        node.out_port(0).get_connection().set_source(reshape_concat.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for shuffle_channel in graph.get_op_nodes(op='ShuffleChannel'):
            self.decompose_shuffle_channel(shuffle_channel)
