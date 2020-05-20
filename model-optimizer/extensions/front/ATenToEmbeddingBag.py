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

from extensions.ops.embedding_bag import EmbeddingBagOffsetsSum, EmbeddingBagPackedSum
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, rename_node


class AtenToEmbeddingBag(FrontReplacementPattern):
    """
    Converts the ATen layer to EmbeddingBag layer.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ATen', operator='embedding_bag'):
            assert node.soft_get('mode') == 0, 'ATen::embedding_bag has unsupported mode, only "sum" ' \
                                               'mode is supported for node {}.'.format(node.id)
            node_name = node.name
            rename_node(node, node_name + '/Old')
            if node.in_port(2).disconnected():
                embedding_bag = EmbeddingBagPackedSum(graph, {'name': node_name}).create_node()
                per_sample_weights_port_id = 2
            else:
                embedding_bag = EmbeddingBagOffsetsSum(graph, {'name': node_name}).create_node()
                per_sample_weights_port_id = 3
                node.in_port(2).get_connection().set_destination(embedding_bag.in_port(2))
            node.in_port(0).get_connection().set_destination(embedding_bag.in_port(0))
            node.in_port(1).get_connection().set_destination(embedding_bag.in_port(1))
            node.out_port(0).get_connection().set_source(embedding_bag.out_port(0))
            if len(node.in_ports()) == 4 and not node.in_port(3).disconnected():
                node.in_port(3).get_connection().set_destination(embedding_bag.in_port(per_sample_weights_port_id))
