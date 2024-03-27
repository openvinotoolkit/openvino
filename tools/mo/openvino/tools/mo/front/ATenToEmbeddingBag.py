# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.embedding_bag import EmbeddingBagOffsetsSum, EmbeddingBagPackedSum
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices, get_canonical_axis_index_node, \
    get_shape_values_by_indices_node


class AtenToEmbeddingBag(FrontReplacementPattern):
    """
    Converts the ATen layer to EmbeddingBag layer.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='ATen', operator='embedding_bag'):
            assert node.soft_get('mode') == 0, 'ATen::embedding_bag has unsupported mode, only "sum" ' \
                                               'mode is supported for node {}.'.format(node.id)
            node_name = node.soft_get('name', node.id)
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

                    weights_shape_node = Shape(graph, {'name': node_name + '/WeightsShape'}).create_node()

                    weights_rank_node = Rank(graph, {'name': node_name + '/WeightsRank'}).create_node()
                    last_dim_node = get_canonical_axis_index_node(weights_rank_node, -1)
                    weights_last_dim = get_shape_values_by_indices_node(weights_shape_node, last_dim_node)

                    weights_first_dim = node_to_get_shape_value_of_indices(weights_shape_node, [0])

                    zero_col_node = create_op_with_const_inputs(graph, Broadcast, {0: int64_array([0])},
                                                                {'name': node_name + '/Broadcast'})
                    zero_col_node.in_port(1).connect(weights_last_dim.out_port(0))

                    default_embeddings_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(0)},
                                                                          {'name': node_name + '/Unsqueeze'})
                    default_embeddings_node.in_port(0).connect(zero_col_node.out_port(0))

                    # expand embedding table with zeros
                    weights_concat = Concat(graph, {'axis': 0, 'in_ports_count': 2,
                                                    'name': node_name + '/Concat'}).create_node()
                    embedding_bag.in_port(0).get_connection().set_destination(weights_concat.in_port(0))
                    weights_concat.in_port(0).get_connection().add_destination(weights_shape_node.in_port(0))
                    weights_concat.in_port(0).get_connection().add_destination(weights_rank_node.in_port(0))
                    weights_concat.in_port(1).connect(default_embeddings_node.out_port(0))
                    weights_concat.out_port(0).connect(embedding_bag.in_port(0))

                    # point default index to expanded part of embedding table
                    weights_first_dim.out_port(0).connect(embedding_bag.in_port(3))
