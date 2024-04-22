# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.roipooling import roipooling_infer
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, \
    undefined_shape_of_rank, set_input_shapes
from openvino.tools.mo.ops.op import Op


class ROIPooling(Op):
    op = 'ROIPooling'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset2',
            'pooled_h': None,
            'pooled_w': None,
            'spatial_scale': 0.0625,
            'method': 'max',
            'infer': roipooling_infer,
            'reverse_infer': self.reverse_infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['pooled_h', 'pooled_w', 'spatial_scale', 'method']

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node, undefined_shape_of_rank(4), shape_array([dynamic_dimension_value, 5]))
