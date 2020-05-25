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

from extensions.ops.embedding_bag import ATenEmbeddingBag
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
            rename_node(node, node_name + '/TBR')
            embedding_bag = ATenEmbeddingBag(graph, {'name': node_name, 'mode': node.soft_get('mode', 1)}).create_node()
            rename_node(embedding_bag, node_name)
            node.in_port(0).get_connection().set_destination(embedding_bag.in_port(0))
            node.in_port(1).get_connection().set_destination(embedding_bag.in_port(1))
            if node.has_port('in', 2) and not node.in_port(2).disconnected():
                node.in_port(2).get_connection().set_destination(embedding_bag.in_port(2))
            if node.has_port('in', 3) and not node.in_port(3).disconnected():
                node.in_port(2).get_connection().set_destination(embedding_bag.in_port(3))
            node.out_port(0).get_connection().set_source(embedding_bag.out_port(0))
