# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.GRUCell import GRUCell
from openvino.tools.mo.ops.concat import Concat

# # POC1
# class GRUBlockCellToGRUCell(FrontReplacementPattern):
#     enabled = True

#     def find_and_replace_pattern(self, graph: Graph):
#         for tf_gru_block_cell in graph.get_op_nodes(op='GRUBlockCell'):
#             original_name = tf_gru_block_cell.soft_get('name', tf_gru_block_cell.id)
#             tf_gru_block_cell['name'] = original_name + '/to_be_removed'

#             new_gru_cell = GRUCell(graph, {}).create_node()
#             rename_node(new_gru_cell, original_name)

#             tf_gru_block_cell.in_port(0).get_connection().set_destination(new_gru_cell.in_port(0))
#             tf_gru_block_cell.in_port(1).get_connection().set_destination(new_gru_cell.in_port(1))

#             concat_w = Concat(graph, {'name': original_name + '/Concat_W',
#                                         'axis': 1}).create_node()

#             concat_w.add_input_port(0)
#             concat_w.add_input_port(1)

#             concat_b = Concat(graph, {'name': original_name + '/Concat_B',
#                                                    'axis': 0}).create_node()

#             concat_b.add_input_port(0)
#             concat_b.add_input_port(1)

#             tf_gru_block_cell.in_port(2).get_connection().set_destination(concat_w.in_port(0))
#             tf_gru_block_cell.in_port(3).get_connection().set_destination(concat_w.in_port(1))

#             tf_gru_block_cell.in_port(4).get_connection().set_destination(concat_b.in_port(0))
#             tf_gru_block_cell.in_port(5).get_connection().set_destination(concat_b.in_port(1))

#             concat_w.out_port(0).connect(new_gru_cell.in_port(2))
#             concat_b.out_port(0).connect(new_gru_cell.in_port(3))

#             tf_gru_block_cell.out_port(3).get_connection().set_source(new_gru_cell.out_port(0))
#             graph.remove_node(tf_gru_block_cell.id)


# ## POC2 (Adjust gate order)
from openvino.tools.mo.ops.split import AttributedSplit

class GRUBlockCellToGRUCell(FrontReplacementPattern):

    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for tf_gru_block_cell in graph.get_op_nodes(op='GRUBlockCell'):
            original_name = tf_gru_block_cell.soft_get('name', tf_gru_block_cell.id)
            tf_gru_block_cell['name'] = original_name + '/to_be_removed'

            new_gru_cell = GRUCell(graph, {}).create_node()
            rename_node(new_gru_cell, original_name)

            tf_gru_block_cell.in_port(0).get_connection().set_destination(new_gru_cell.in_port(0))
            tf_gru_block_cell.in_port(1).get_connection().set_destination(new_gru_cell.in_port(1))

            concat_w = Concat(graph, {'name': original_name + '/Concat_W',
                                        'axis': 1}).create_node()

            concat_w.add_input_port(0)
            concat_w.add_input_port(1)


            concat_b = Concat(graph, {'name': original_name + '/Concat_B',
                                                   'axis': 0}).create_node()

            concat_b.add_input_port(0)
            concat_b.add_input_port(1)

            tf_gru_block_cell.in_port(2).get_connection().set_destination(concat_w.in_port(0))
            tf_gru_block_cell.in_port(3).get_connection().set_destination(concat_w.in_port(1))

            tf_gru_block_cell.in_port(4).get_connection().set_destination(concat_b.in_port(0))
            tf_gru_block_cell.in_port(5).get_connection().set_destination(concat_b.in_port(1))

            # W
            # Convert gate order "rzh" -> "zrh"
            split_rzh_w = AttributedSplit(graph, {'name': original_name + '/Split_rzh_w_W', 'axis': 1, 'num_splits': 3}).create_node()

            split_rzh_w.out_port(1)
            concat_zrh_w = Concat(graph, {'name': original_name + '/Concat_zrh_w_W',
                                                   'axis': 1}).create_node()
            concat_zrh_w.add_input_port(0)
            concat_zrh_w.add_input_port(1)
            concat_zrh_w.add_input_port(2)

            # r at 0 -> r at 1
            split_rzh_w.out_port(0).connect(concat_zrh_w.in_port(1))

            # z at 1 -> z at 0
            split_rzh_w.out_port(1).connect(concat_zrh_w.in_port(0))

            # h at 2 -> h at 2
            split_rzh_w.out_port(2).connect(concat_zrh_w.in_port(2))

            concat_w.out_port(0).connect(split_rzh_w.in_port(0))

            # B
            # Convert gate order "rzh" -> "zrh"
            split_rzh_b = AttributedSplit(graph, {'name': original_name + '/Split_rzh_B', 'axis': 0, 'num_splits': 3}).create_node()

            split_rzh_b.out_port(1)
            concat_zrh_b = Concat(graph, {'name': original_name + '/Concat_zrh_B',
                                                   'axis': 0}).create_node()
            concat_zrh_b.add_input_port(0)
            concat_zrh_b.add_input_port(1)
            concat_zrh_b.add_input_port(2)

            # r at 0 -> r at 1
            split_rzh_b.out_port(0).connect(concat_zrh_b.in_port(1))

            # z at 1 -> z at 0
            split_rzh_b.out_port(1).connect(concat_zrh_b.in_port(0))

            # h at 2 -> h at 2
            split_rzh_b.out_port(2).connect(concat_zrh_b.in_port(2))
            concat_b.out_port(0).connect(split_rzh_b.in_port(0))


            concat_zrh_w.out_port(0).connect(new_gru_cell.in_port(2))
            concat_zrh_b.out_port(0).connect(new_gru_cell.in_port(3))

            tf_gru_block_cell.out_port(3).get_connection().set_source(new_gru_cell.out_port(0))
            graph.remove_node(tf_gru_block_cell.id)


