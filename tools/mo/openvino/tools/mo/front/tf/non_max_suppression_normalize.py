# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.crop import Crop
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class TFNonMaxSuppressionNormalize(FrontReplacementSubgraph):
    """
    The inputs and outputs format of the TF implementation of the NMS layer is different from the OpenVINO
    implementation and supports just one batch and image class. This transformation converts inputs and outputs to
    match the OpenVINO implementation.

    TF inputs: boxes = [num_boxes, 4]
               scores = [num_boxes]
       outputs: box_indices [selected_boxes_count]
                box_scores [selected_boxes_count]
                valid_outputs selected_boxes_count

    OV inputs: boxes = [num_batches, num_boxes, 4]
               scores = [num_batches, num_classes, num_boxes]
       outputs: selected_indices [num_selected_indices, 3] where each element is [batch_index, class_index, box_index]
                selected_scores [num_selected_indices, 3] where each element is [batch_index, class_index, box_score]
                valid_outputs num_selected_indices
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.non_max_suppression_normalize import NonMaxSuppressionNormalize
        return [NonMaxSuppressionNormalize]

    def find_and_replace_pattern(self, graph: Graph):
        for nms in graph.get_op_nodes(op='NonMaxSuppression'):
            # prepare inputs to the NonMaximumSuppression Node
            unsqueeze_boxes = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                               {'name': nms.soft_get('name') + '/Unsqueeze_0'})
            nms.in_port(0).get_connection().insert_node(unsqueeze_boxes)

            unsqueeze_box_scores = create_op_node_with_second_input(graph, Reshape, int64_array([1, 1, -1]),
                                                                    {'name': nms.soft_get('name') + '/Unsqueeze_1'})
            nms.in_port(1).get_connection().insert_node(unsqueeze_box_scores)

            nms_name = nms.soft_get('name', nms.id)

            # prepare output #0
            crop_box_indices_name = nms_name + '/Crop_boxes_'
            crop_box_indices = Crop(graph, {'name': crop_box_indices_name, 'axis': int64_array([1]),
                                            'offset': int64_array([2]), 'dim': int64_array([1])}).create_node()
            nms.out_port(0).get_connection().insert_node(crop_box_indices)
            squeeze_output_boxes = create_op_node_with_second_input(graph, Squeeze, int64_array([1]),
                                                                    {'name': crop_box_indices_name + '/Squeeze'})
            crop_box_indices.out_port(0).get_connection().insert_node(squeeze_output_boxes)

            num_of_outputs = len([port for port in nms.out_ports().values() if not port.disconnected()])

            if num_of_outputs == 1:
                continue

            # prepare output #1
            crop_score_indices_name = nms_name + '/Crop_scores_'
            crop_score_indices = Crop(graph, {'name': crop_score_indices_name, 'axis': int64_array([1]),
                                              'offset': int64_array([2]), 'dim': int64_array([1])}).create_node()
            nms.out_port(1).get_connection().insert_node(crop_score_indices)
            squeeze_output_scores = create_op_node_with_second_input(graph, Squeeze, int64_array([1]),
                                                                     {'name': crop_score_indices_name + '/Squeeze'})
            crop_score_indices.out_port(0).get_connection().insert_node(squeeze_output_scores)
