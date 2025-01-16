# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, set_input_shapes
from openvino.tools.mo.ops.op import Op


class ExperimentalDetectronROIFeatureExtractor(Op):
    op = 'ExperimentalDetectronROIFeatureExtractor'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=self.op,
            op=self.op,
            version='opset6',
            infer=self.infer,
            reverse_infer=self.reverse_infer,
            out_ports_count=2,
        )

        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            ('pyramid_scales', lambda node: ','.join(map(str, node['pyramid_scales']))),
            'output_size',
            'sampling_ratio',
            ('aligned', lambda node: str(bool(node.soft_get('aligned', False))).lower())]

    @staticmethod
    def infer(node):
        input_rois_shape = node.in_port(0).data.get_shape()
        rois_num = input_rois_shape[0]
        input_features_level_0_shape = node.in_port(1).data.get_shape()
        channels_num = input_features_level_0_shape[1]
        node.out_port(0).data.set_shape([rois_num, channels_num, node.output_size, node.output_size])
        if not node.out_port(1).disconnected():
            node.out_port(1).data.set_shape([rois_num, 4])

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node,
                         shape_array([dynamic_dimension_value, 4]),
                         shape_array([1, dynamic_dimension_value, dynamic_dimension_value, dynamic_dimension_value]))
