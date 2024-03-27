# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Sub
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.pad import Pad
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class BatchToSpaceNormalizer(FrontReplacementPattern):
    """
    This transformation converts BatchToSpace, SpaceToBatch operations (TensorFlow semantic)
    to BatchToSpace, SpaceToBatch operations (OpenVINO semantic).
    Refer to the Op implementation for the operations semantics description.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.rank_decomposer import RankDecomposer
        return [RankDecomposer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='SpaceToBatch') + graph.get_op_nodes(op='BatchToSpace'):
            node.add_input_port(3, skip_if_exist=True)

            # convert TF representation of the pads/crops as [N, 2] to OV representation: [N] and [N]
            transposed_pads = create_op_with_const_inputs(graph, Transpose, {1: int64_array([1, 0])})
            node.in_port(2).get_connection().set_destination(transposed_pads.in_port(0))
            split_pads = create_op_with_const_inputs(graph, Split, {1: int64_array(0)}, {'num_splits': 2})
            transposed_pads.out_port(0).connect(split_pads.in_port(0))
            for port_ind in range(2):
                node.in_port(port_ind + 2).connect(split_pads.out_port(port_ind))
                node.in_port(port_ind + 2).get_connection().insert_node(
                    create_op_with_const_inputs(graph, Squeeze, {1: int64_array([0])}))

            # add zeros/ones to related inputs to align it with data input
            in0_rank = Rank(graph, {'name': node.name + '/rank_0'}).create_node()
            in1_shape = Shape(graph, {'name': node.name + '/rank_1'}).create_node()

            diff_size = Sub(graph, {'name': node.name + '/sub_0'}).create_node()
            diff = Sub(graph, {'name': node.name + '/sub_1'}).create_node()
            const_begin = Const(graph, {'value': int64_array([1])}).create_node()
            const_pad_val = Const(graph, {'value': int64_array(1)}).create_node()

            block_shape = Pad(graph, {'name': node.name + '/aligned_block_shape', 'mode': 'constant'}).create_node()

            # in case of SpaceToBatch begin = pads_begin, end = pads_end
            # in case of BatchToSpace begin = crops_begin, end = crops_end
            new_begin_name = '/aligned_pads_begin'
            new_end_name = '/aligned_pads_end'
            if node.type == 'BatchToSpace':
                new_begin_name = '/aligned_crops_begin'
                new_end_name = '/aligned_crops_end'

            begin = Pad(graph, {'name': node.name + new_begin_name, 'mode': 'constant'}).create_node()
            end = Pad(graph, {'name': node.name + new_end_name, 'mode': 'constant'}).create_node()

            in0_rank_1d = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                           {'name': node.name + '/1d_rank_of_0'}, in0_rank)

            node.in_port(0).get_source().connect(in0_rank.in_port(0))
            node.in_port(1).get_source().connect(in1_shape.in_port(0))
            in0_rank_1d.out_port(0).connect(diff_size.in_port(0))
            in1_shape.out_port(0).connect(diff_size.in_port(1))
            diff_size.out_port(0).connect(diff.in_port(0))
            const_begin.out_port(0).connect(diff.in_port(1))
            const_pad_val.out_port(0).connect(block_shape.in_port(3))

            inputs_array = [block_shape, begin, end]
            for idx, input_to_node in enumerate(inputs_array):
                name_of_input_to_node = input_to_node.name
                node.in_port(idx + 1).get_connection().set_destination(input_to_node.in_port(0))
                const_begin.out_port(0).connect(input_to_node.in_port(1))
                diff.out_port(0).connect(input_to_node.in_port(2))
                input_to_node.out_port(0).connect(node.in_port(idx + 1))
                convert = Cast(graph, {'name': name_of_input_to_node + '/i64', 'dst_type': np.int64}).create_node()
                input_to_node.in_port(0).get_connection().insert_node(convert)
