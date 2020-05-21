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

from extensions.ops.embedding_bag import EmbeddingBagOffsetsSum, EmbeddingBagPackedSum
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const


class AtenToEmbeddingBag(MiddleReplacementPattern):
    """
    Converts the ATen layer to EmbeddingBag layer.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ATen', operator='embedding_bag'):
            assert node.soft_get('mode') == 0, 'ATen::embedding_bag has unsupported mode, only "sum" ' \
                                               'mode is supported for node {}.'.format(node.id)
            node_name = node.name
            rename_node(node, node_name + '/TBR')
            is_packed = False
            if len(node.in_ports()) < 3 or node.in_port(2).disconnected():
                is_packed = True
                embedding_bag = EmbeddingBagPackedSum(graph, {'name': node_name}).create_node()
            else:
                embedding_bag = EmbeddingBagOffsetsSum(graph, {'name': node_name}).create_node()
                node.in_port(2).get_connection().set_destination(embedding_bag.in_port(2))
            rename_node(embedding_bag, node_name)
            node.in_port(0).get_connection().set_destination(embedding_bag.in_port(0))
            node.in_port(1).get_connection().set_destination(embedding_bag.in_port(1))
            node.out_port(0).get_connection().set_source(embedding_bag.out_port(0))
            if len(node.in_ports()) == 4 and not node.in_port(3).disconnected():
                if is_packed:
                    node.in_port(3).get_connection().set_destination(embedding_bag.in_port(2))
                else:
                    # connect per_sample_weights
                    node.in_port(3).get_connection().set_destination(embedding_bag.in_port(4))

                    weights_shape = embedding_bag.in_port(0).data.get_shape()

                    # expand embedding table with zeros
                    default_embeddings = np.zeros([1, weights_shape[-1]])
                    weights_concat = create_op_with_const_inputs(graph, Concat, {1: default_embeddings},
                                                                 {'axis': 0, 'in_ports_count': 2})
                    embedding_bag.in_port(0).get_connection().set_destination(weights_concat.in_port(0))
                    weights_concat.out_port(0).connect(embedding_bag.in_port(0))

                    # point default index to expanded part of embedding table
                    default_index = Const(graph, {'value': int64_array(weights_shape[0])}).create_node()
                    default_index.out_port(0).connect(embedding_bag.in_port(3))
