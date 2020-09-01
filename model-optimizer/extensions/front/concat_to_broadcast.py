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

from extensions.ops.elementwise import Pow
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_node
from mo.ops.broadcast import Broadcast
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_shape_value_of_indices


class ConcatToBroadcast(FrontReplacementPattern):
    enabled = True

    @staticmethod
    def concat_to_broadcast(concat: Node):
        assert concat.soft_get('type') == 'Concat'
        data_inputs = list((concat.in_ports().keys()))
        axis = None  # TODO: should be removed after Concat operation normalization
        if concat.has_valid('axis'):
            axis = concat.axis
        if concat.has_valid('N'):
            data_inputs.remove(concat.N)
            axis_node = concat.in_port(concat.N).get_source().node
            if axis_node.has_valid('value'):
                axis = axis_node.value.item(0)
        if axis is None or axis < 0 or len(data_inputs) < 2:
            return
        sample_port = concat.in_port(data_inputs[0]).get_source()
        for idx in data_inputs:
            if not concat.in_port(idx).get_source() == sample_port:
                return
        graph = concat.graph
        name = concat.soft_get('name', concat.id)
        sample_node_name = sample_port.node.soft_get('name', sample_port.node.id)
        rename_node(node=concat, name=name + '/to_be_removed')

        shape = Shape(graph, {'name': sample_node_name + '/out_' + sample_port.idx + '/shape'}).create_node()
        shape.in_port(0).connect(sample_port)



        broadcast = Broadcast(graph, {'name': name}).create_node()
        rename_node(broadcast, name)

        div.in_port(0).get_connection().set_destination(mul.in_port(0))
        div.in_port(1).get_connection().set_destination(mul.in_port(1))
        div.out_port(0).get_connection().set_source(mul.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for concat in graph.get_op_nodes(op='Concat'):
            self.concat_to_broadcast(concat)
