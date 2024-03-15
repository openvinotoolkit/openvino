# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.GRUCell import GRUCell
from openvino.tools.mo.ops.split import AttributedSplit
from openvino.tools.mo.ops.transpose import Transpose


class GRUBlockCellToGRUCell(FrontReplacementPattern):
    """
    This transformation converts TF GRUBlockCell to mo.ops.GRUCell
    by alignment of weights and bias inputs.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for tf_gru_block_cell in graph.get_op_nodes(op='GRUBlockCell'):
            original_name = tf_gru_block_cell.soft_get('name', tf_gru_block_cell.id)
            new_gru_cell = GRUCell(graph, {}).create_node()
            rename_nodes([(tf_gru_block_cell, original_name + '/to_be_removed'), (new_gru_cell, original_name)])

            # Connect X data port
            tf_gru_block_cell.in_port(0).get_connection().set_destination(new_gru_cell.in_port(0))
            # Connect hidden state port
            tf_gru_block_cell.in_port(1).get_connection().set_destination(new_gru_cell.in_port(1))

            # W (Weights)
            # z - update, r - reset, h - hidden
            # Convert gate order W_rz, W_h -> W_zrh
            split_rz_w = AttributedSplit(graph, {'name': original_name + '/Split_W_rz', 'axis': 1, 'num_splits': 2}).create_node()

            # Split W_rz to W_r and W_z
            tf_gru_block_cell.in_port(2).get_connection().set_destination(split_rz_w.in_port(0))

            concat_zrh_w = Concat(graph, {'name': original_name + '/Concat_W_zrh', 'in_ports_count': 3,
                                          'axis': 1}).create_node()

            # Swap and concat gates: W_rz -> W_zr
            split_rz_w.out_port(0).connect(concat_zrh_w.in_port(1))
            split_rz_w.out_port(1).connect(concat_zrh_w.in_port(0))

            # Conncat W_h gate: W_zr -> W_zrh
            tf_gru_block_cell.in_port(3).get_connection().set_destination(concat_zrh_w.in_port(2))

            # B (Bias)
            # z - update, r - reset, h - hidden
            # Convert gate order B_rz, B_h -> B_zrh
            split_rz_b = AttributedSplit(graph, {'name': original_name + '/Split_B_rz', 'axis': 0, 'num_splits': 2}).create_node()

            # Split B_rz to B_r and B_z
            tf_gru_block_cell.in_port(4).get_connection().set_destination(split_rz_b.in_port(0))

            concat_zrh_b = Concat(graph, {'name': original_name + '/Concat_B_zrh', 'in_ports_count': 3,
                                          'axis': 0}).create_node()

            # Swap and concat gates: B_rz -> B_zr
            split_rz_b.out_port(0).connect(concat_zrh_b.in_port(1))
            split_rz_b.out_port(1).connect(concat_zrh_b.in_port(0))

            # Concat B_h gate: B_zr -> B_zrh
            tf_gru_block_cell.in_port(5).get_connection().set_destination(concat_zrh_b.in_port(2))

            # Transpose W Shape [input_size + hidden_size, 3 * hidden_size] to [3 * hidden_size, input_size + hidden_size]
            permute_order = int64_array([1, 0])
            transpose_w = create_op_node_with_second_input(graph, Transpose, permute_order,
                                                          dict(name=original_name + '/Transpose_W'), concat_zrh_w)

            transpose_w.out_port(0).connect(new_gru_cell.in_port(2))
            concat_zrh_b.out_port(0).connect(new_gru_cell.in_port(3))

            tf_gru_block_cell.out_port(3).get_connection().set_source(new_gru_cell.out_port(0))
            graph.remove_nodes_from([tf_gru_block_cell.id])
