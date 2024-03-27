# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, set_input_shapes
from openvino.tools.mo.ops.op import Op


class ExperimentalDetectronDetectionOutput(Op):
    op = 'ExperimentalDetectronDetectionOutput'
    enabled = True

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=self.op,
            op=self.op,
            version='opset6',
            infer=self.infer,
            reverse_infer=self.reverse_infer,
            type_infer=self.type_infer,
            in_ports_count=4,
            out_ports_count=3,
        )

        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            ('class_agnostic_box_regression', lambda node: str(bool(node['class_agnostic_box_regression'])).lower()),
            'max_detections_per_image',
            'nms_threshold',
            'num_classes',
            'post_nms_count',
            'score_threshold',
            'max_delta_log_wh',
            ('deltas_weights', lambda node: ','.join(map(str, node['deltas_weights'])))]

    @staticmethod
    def infer(node):
        rois_num = node.max_detections_per_image
        # boxes
        node.out_port(0).data.set_shape([rois_num, 4])
        # classes, scores, batch indices
        # We use range(1, 1 + max(node.out_ports().keys())) instead of range(1, 3), because there are incorrectly
        # generated models where ExperimentalDetectronDetectionOutput has 4 outputs.
        for port_ind in range(1, 1 + max(node.out_ports().keys())):
            if not node.out_port(port_ind).disconnected():
                node.out_port(port_ind).data.set_shape([rois_num])

    @staticmethod
    def type_infer(node):
        in_data_type = node.in_port(0).get_data_type()
        node.out_port(0).set_data_type(in_data_type)
        node.out_port(1).set_data_type(np.int32)  # the second output contains class indices
        node.out_port(2).set_data_type(in_data_type)
        if node.is_out_port_connected(3):
            node.out_port(3).set_data_type(np.int32)  # the fourth output contains batch indices

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node,
                         shape_array([dynamic_dimension_value, 4]),
                         shape_array([dynamic_dimension_value, node['num_classes'] * 4]),
                         shape_array([dynamic_dimension_value, node['num_classes']]),
                         shape_array([1, 3]))
