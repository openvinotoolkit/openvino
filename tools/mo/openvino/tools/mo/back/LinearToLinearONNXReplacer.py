# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.InterpolateReshape import InterpolateConcat, InterpolateReshapeWA
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class LinearToLinearONNXReplacer(BackReplacementPattern):
    """
    If we don't use this transformation, then we have a performance drop, because CPU and GPU have no optimized
    version of the 'linear' mode of the operation Interpolate.
    TODO: delete this transformation, when CPU and GPU will have optimized version of the 'linear' mode.
    """
    enabled = True

    def run_after(self):
        return [InterpolateConcat, InterpolateReshapeWA]

    def find_and_replace_pattern(self, graph: Graph):
        for interpolate_node in graph.get_op_nodes(type='Interpolate', version='opset4', mode='linear'):
            input_shape = interpolate_node.in_port(0).data.get_shape()
            interpolate_name = interpolate_node.soft_get('name', interpolate_node.id)
            assert input_shape is not None, \
                'Shape of interpolated data for node {} must not be None'.format(interpolate_name)
            input_rank = len(input_shape)
            if input_rank == 4:
                interpolate_node['mode'] = 'linear_onnx'
