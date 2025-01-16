# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, set_input_shapes
from openvino.tools.mo.ops.op import Op


class ExperimentalDetectronTopKROIs(Op):
    op = 'ExperimentalDetectronTopKROIs'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=self.op,
            op=self.op,
            version='experimental',
            reverse_infer=self.reverse_infer,
            infer=self.infer
        )
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['max_rois', ]

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape([node.max_rois, 4])

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node, shape_array([dynamic_dimension_value, 4]), shape_array([dynamic_dimension_value]))
