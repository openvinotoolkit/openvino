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

from extensions.ops.embedding_bag import EmbeddingBag
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, rename_node


class AtenToEmbeddingBag(FrontReplacementPattern):
    """
    Converts the ATen layer to EmbeddingBag layer.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ATen', operator='embedding_bag'):
            node_name = node.name
            rename_node(node, node_name + '/Old')
            embedding_bag = EmbeddingBag(graph, {'name': node_name, 'mode': node.mode,
                                                 'scale_grad_by_freq': node.scale_grad_by_freq}).create_node()
            node.in_port(0).get_connection().set_destination(embedding_bag.in_port(0))
            node.in_port(1).get_connection().set_destination(embedding_bag.in_port(1))
            node.in_port(2).get_connection().set_destination(embedding_bag.in_port(2))
            node.out_port(0).get_connection().set_source(embedding_bag.out_port(0))
