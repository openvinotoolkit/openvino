# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, \
    undefined_shape_of_rank, set_input_shapes
from openvino.tools.mo.ops.op import Op


class ExperimentalDetectronGenerateProposalsSingleImage(Op):
    op = 'ExperimentalDetectronGenerateProposalsSingleImage'

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
        return [
            'min_size',
            'nms_threshold',
            'post_nms_count',
            'pre_nms_count'
        ]

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape([node.post_nms_count, 4])
        node.out_port(1).data.set_shape([node.post_nms_count])

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node,
                         undefined_shape_of_rank(1),
                         shape_array([dynamic_dimension_value, 4]),
                         undefined_shape_of_rank(3),
                         undefined_shape_of_rank(3))
