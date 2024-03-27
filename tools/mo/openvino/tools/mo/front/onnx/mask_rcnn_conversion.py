# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.onnx.softmaxONNX_to_softmax import SoftmaxONNXFrontReplacer
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.detection_output_onnx import ExperimentalDetectronDetectionOutput
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.roifeatureextractor_onnx import ExperimentalDetectronROIFeatureExtractor
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.reshape import Reshape


class ONNXMaskRCNNTransformation(FrontReplacementFromConfigFileGeneral):
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

    def run_before(self):
        # the class_predicitons_node which is used in this transformation is of op SoftMaxONNX. But operations of op SoftMaxONNX
        # will be replaced with a transformation SoftmaxONNXFrontReplacer
        return [SoftmaxONNXFrontReplacer]

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        insert_ExperimentalDetectronROIFeatureExtractor2(graph, replacement_descriptions)
        insert_do(graph, replacement_descriptions)
        insert_ExperimentalDetectronROIFeatureExtractor1(graph, replacement_descriptions)


def insert_do(graph: Graph, replacement_descriptions: dict):
    do_outputs = replacement_descriptions['do_outputs']
    prior_boxes_node = Node(graph, 'ROIFeatureExtractor_2')
    num_classes = 81
    box_regressions_input_node = Node(graph, replacement_descriptions['box_regressions_input_node'])
    box_regressions_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 4 * num_classes]),
                                                            dict(name='box_regressions'), box_regressions_input_node)

    class_predicitons_node = Node(graph, replacement_descriptions['class_predicitons_node'])
    im_info_node = Parameter(graph, {"name": 'im_info', 'shape': int64_array([1, 3])}).create_node()

    do_node = ExperimentalDetectronDetectionOutput(graph, {'name': 'DetectionOutput',
                                                           'class_agnostic_box_regression': 0,
                                                           'deltas_weights': mo_array([10.0, 10.0, 5.0, 5.0]),
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
    # the consumer of the second output port of the ExperimentalDetectronDetectionOutput is the Mul node which second
    # input is of type int64 so it is necessary to insert Cast to have data types match
    do_node.out_port(1).get_connection().insert_node(Cast(graph, {'dst_type': np.int64}).create_node())


def insert_ExperimentalDetectronROIFeatureExtractor1(graph: Graph, replacement_descriptions: dict):
    if 'ROIFeatureExtractor1_output' not in replacement_descriptions:
        # In case of Faster-RCNN this transformation is not needed and this attribute shouldn't be set
        return
    input_fpn_heads = replacement_descriptions['input_fpn_heads']
    old_output_node = Node(graph, replacement_descriptions['ROIFeatureExtractor1_output'])
    input_fpn_head_nodes = [Node(graph, node_id) for node_id in input_fpn_heads]
    fpn_roi_align = ExperimentalDetectronROIFeatureExtractor(graph, {'name': 'ROIFeatureExtractor_1',
                                                                     'output_size': 14,
                                                                     'pyramid_scales': int64_array(
                                                                         [4, 8, 16, 32, 64]),
                                                                     'sampling_ratio': 2,
                                                                     'in_ports_count': 5}).create_node()
    fpn_roi_align.in_port(0).connect(Node(graph, 'DetectionOutput').out_port(0))
    for ind, fpn_node in enumerate(input_fpn_head_nodes):
        fpn_roi_align.in_port(ind + 1).connect(fpn_node.out_port(0))

    old_output_node.out_port(0).get_connection().set_source(fpn_roi_align.out_port(0))


def insert_ExperimentalDetectronROIFeatureExtractor2(graph: Graph, replacement_descriptions: dict):
    input_fpn_heads = replacement_descriptions['input_fpn_heads']
    old_output_node = Node(graph, replacement_descriptions['ROIFeatureExtractor2_output'])
    input_fpn_head_nodes = [Node(graph, node_id) for node_id in input_fpn_heads]
    fpn_roi_align = ExperimentalDetectronROIFeatureExtractor(graph, {'name': 'ROIFeatureExtractor_2',
                                                                     'output_size': 7,
                                                                     'pyramid_scales': int64_array(
                                                                         [4, 8, 16, 32, 64]),
                                                                     'sampling_ratio': 2,
                                                                     'in_ports_count': 5}).create_node()
    fpn_roi_align.in_port(0).connect(Node(graph, replacement_descriptions['ROIFeatureExtractor2_input']).out_port(0))
    for ind, fpn_node in enumerate(input_fpn_head_nodes):
        fpn_roi_align.in_port(ind + 1).connect(fpn_node.out_port(0))

    old_output_node.out_port(0).get_connection().set_source(fpn_roi_align.out_port(0))
