# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, is_fully_defined, shape_insert
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class ExpandDims(Op):
    """
    The ExpandDims layer adds dimensions with shape 1 to the specified positions. The positions is a layer attribute,
    not a separate input.
    """
    op = 'ExpandDims'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'reinterp_shape': True,
            'infer': self.infer,
            'expand_axis': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        if input_shape is None:
            raise Error('Input shape for node "{}" is None'.format(node_name))

        assert len(node.in_nodes()) == 1, 'Wrong number of inputs to the layer {}'.format(node_name)

        if not node.has_valid('expand_axis'):
            raise Error('ExpandDims axis is not defined for node {}'.format(node_name))

        expand_axes = node.expand_axis
        if expand_axes is None:
            raise Error('The "expand_axis" attribute is None for node "{}"'.format(node_name))

        if isinstance(expand_axes, int):
            expand_axes = int64_array([expand_axes])
        elif expand_axes.ndim == 0:
            expand_axes = expand_axes.reshape([1])

        # expand_axis is a position where the new axis is placed so expand_dims works for negative axis in a different
        # way not as insert operation
        for expand_axis in expand_axes:
            if expand_axis < 0:
                expand_axis += len(input_shape) + 1

        expand_axes = sorted(expand_axes)
        output_shape = input_shape.copy()
        for expand_axis in expand_axes:
            output_shape = shape_insert(output_shape, expand_axis, 1)

        if input_value is not None and is_fully_defined(output_shape):
            node.out_port(0).data.set_value(input_value.reshape(output_shape))
        else:
            node.out_port(0).data.set_shape(output_shape)
