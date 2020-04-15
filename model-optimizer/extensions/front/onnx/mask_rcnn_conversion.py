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

from extensions.ops.detectionoutput_onnx import ExperimentalDetectronDetectionOutput
from extensions.ops.parameter import Parameter
from extensions.ops.roifeatureextractor_onnx import ExperimentalDetectronROIFeatureExtractor
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph
from mo.graph.graph import Node
from mo.ops.reshape import Reshape

input_fpn_heads = ('486', '454', '422', '390')


class ObjectDetectionAPIOutputReplacement(FrontReplacementFromConfigFileGeneral):
    """
    This transformation performs 3 actions:
    1. Replaces a sub-graph calculating ROIAlign over FPN heads with a single ExperimentalDetectronROIFeatureExtractor
    node.
    2. Replaces a sub-graph calculating post-processing of background/foreground with a single
    ExperimentalDetectronDetectionOutput node.
    3. Replaces another sub-graph calculating ROIAlign over FPN heads with a single
    ExperimentalDetectronROIFeatureExtractor node. These ROIAligns get boxes from the
    ExperimentalDetectronDetectionOutput node.
    """
    replacement_id = 'ONNXMaskRCNNReplacement'

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        insert_ExperimentalDetectronROIFeatureExtractor2(graph)
        insert_do(graph, replacement_descriptions)
        insert_ExperimentalDetectronROIFeatureExtractor1(graph)


def insert_do(graph: Graph, replacement_descriptions):
    do_outputs = ['6530', '6532', '6534']
    prior_boxes_node = Node(graph, 'ROIFeatureExtractor_2')
    num_classes = 81
    box_regressions_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4 * num_classes]),
                                                            dict(name='box_regressions'), Node(graph, '2773'))

    class_predicitons_node = Node(graph, '2774')
    im_info_node = Parameter(graph, {"name": 'im_info', 'shape': int64_array([1, 3])}).create_node()

    do_node = ExperimentalDetectronDetectionOutput(graph, {'name': 'DetectionOutput',
                                                           'class_agnostic_box_regression': 0,
                                                           'deltas_weights': np.array([10.0, 10.0, 5.0, 5.0]),
                                                           'max_delta_log_wh':
                                                               replacement_descriptions['max_delta_log_wh'],
                                                           'nms_threshold': replacement_descriptions['nms_threshold'],
                                                           'score_threshold':
                                                               replacement_descriptions['score_threshold'],
                                                           'num_classes': num_classes,
                                                           'max_detections_per_image':
                                                               replacement_descriptions['max_detections_per_image'],
                                                           'post_nms_count': replacement_descriptions['post_nms_count']
                                                           }).create_node()
    prior_boxes_node.out_port(1).connect(do_node.in_port(0))
    box_regressions_node.out_port(0).connect(do_node.in_port(1))
    class_predicitons_node.out_port(0).connect(do_node.in_port(2))
    im_info_node.out_port(0).connect(do_node.in_port(3))

    do_output_ports = [do_node.out_port(0), do_node.out_port(1), do_node.out_port(2)]
    old_do_output_nodes = [Node(graph, node_id) for node_id in do_outputs]
    for old_node, new_port in zip(old_do_output_nodes, do_output_ports):
        old_node.out_port(0).get_connection().set_source(new_port)


def insert_ExperimentalDetectronROIFeatureExtractor1(graph: Graph):
    old_output_node = Node(graph, '6795')
    input_fpn_head_nodes = [Node(graph, node_id) for node_id in input_fpn_heads]
    fpn_roi_align = ExperimentalDetectronROIFeatureExtractor(graph, {'name': 'ROIFeatureExtractor_1',
                                                                     'distribute_rois_between_levels': 1,
                                                                     'image_id': 0,
                                                                     'output_size': 14,
                                                                     'preserve_rois_order': 1,
                                                                     'pyramid_scales': int64_array(
                                                                         [4, 8, 16, 32, 64]),
                                                                     'sampling_ratio': 2, }).create_node()
    fpn_roi_align.in_port(0).connect(Node(graph, 'DetectionOutput').out_port(0))
    for ind, fpn_node in enumerate(input_fpn_head_nodes):
        fpn_roi_align.in_port(ind + 1).connect(fpn_node.out_port(0))

    old_output_node.out_port(0).get_connection().set_source(fpn_roi_align.out_port(0))


def insert_ExperimentalDetectronROIFeatureExtractor2(graph: Graph):
    old_output_node = Node(graph, '2751')
    input_fpn_head_nodes = [Node(graph, node_id) for node_id in input_fpn_heads]
    fpn_roi_align = ExperimentalDetectronROIFeatureExtractor(graph, {'name': 'ROIFeatureExtractor_2',
                                                                     'distribute_rois_between_levels': 1,
                                                                     'image_id': 0,
                                                                     'output_size': 7,
                                                                     'preserve_rois_order': 1,
                                                                     'pyramid_scales': int64_array(
                                                                         [4, 8, 16, 32, 64]),
                                                                     'sampling_ratio': 2, }).create_node()
    fpn_roi_align.in_port(0).connect(Node(graph, '2490').out_port(0))
    for ind, fpn_node in enumerate(input_fpn_head_nodes):
        fpn_roi_align.in_port(ind + 1).connect(fpn_node.out_port(0))

    old_output_node.out_port(0).get_connection().set_source(fpn_roi_align.out_port(0))
