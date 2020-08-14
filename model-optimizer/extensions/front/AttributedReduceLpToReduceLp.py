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

from extensions.ops.ReduceOps import ReduceLp
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_node


class AttributedReduceLpToReduceLp(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        # We need to add normalization order value for ReduceLp operation as 3rd input
        for node in graph.get_op_nodes(op='AttributedReduceLp'):
            name = node.soft_get('name', node.id)

            p = node.soft_get('p', None)
            assert p is not None, 'AttributedReduceLp must have "p" attribute defined for node {}'.format(name)
            keep_dims = node.soft_get('keep_dims', None)
            assert keep_dims is not None, 'AttributedReduceLp must have "keep_dims" defined for node {}'.format(name)

            reduce_node = create_op_with_const_inputs(graph, ReduceLp, {2: p}, {'keep_dims': keep_dims})

            node.in_port(0).get_connection().set_destination(reduce_node.in_port(0))
            node.in_port(1).get_connection().set_destination(reduce_node.in_port(1))
            node.out_port(0).get_connection().set_source(reduce_node.out_port(0))

            graph.remove_node(node.id)
            rename_node(reduce_node, name)
