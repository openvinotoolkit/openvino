# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import add_convolution_to_swap_xy_coordinates, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class CropAndResizeReplacement(FrontReplacementOp):
    """
    The CropAndResize operation from TF gets separate input with boxes coordinates and image batch indices. But
    ROIPooling operation in the OpenVINO receives them as a single concatenated input. This replacer
    concatenates two inputs into a new one.
    """
    op = "CropAndResize"
    enabled = True

    def nodes_to_remove(self, graph: Graph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: Graph, node: Node):
        if node.has_and_set('inputs_preprocessed'):
            log.debug('Node "{}" has already been preprocessed'.format(node.soft_get('name')))
            return []
        # reshape tensor with batch indices to 2d
        unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([1]),
                                                          {'name': node.name + '/Unsqueeze'}, node.in_node(2))

        convert_node = Cast(graph, {'name': unsqueeze_node.name + '/ToFloat',
                                    'dst_type': data_type_str_to_np(graph.graph['cmd_params'].data_type)}).create_node()

        convert_node.in_port(0).connect(unsqueeze_node.out_port(0))

        concat_op = Concat(graph, {'axis': 1, 'name': node.name + '/concat_batch_indices_and_boxes',
                                   'in_ports_count': 2})
        concat_node = concat_op.create_node([convert_node, node.in_node(1)])

        # do not remove edge with crop_size because it is needed in the partial infer
        graph.remove_edge(node.in_node(1).id, node.id)

        # input to the CropAndResize contains boxes coordinates in YXYX layout. But OV layer ROIPooling expects
        # coordinates in the XYXY layout, so convolution is added here to swap coordinates
        swapped_box_coordinates_node = add_convolution_to_swap_xy_coordinates(graph, concat_node, 5)

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                           dict(name=swapped_box_coordinates_node.id + '/reshape_2d_'),
                                                           swapped_box_coordinates_node)
        graph.create_edge(reshape_2d_node, node, 0, 1)

        # do not replace any output edge
        return []
