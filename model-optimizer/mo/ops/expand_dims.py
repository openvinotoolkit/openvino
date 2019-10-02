"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error


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
            'op': __class__.op,
            'reinterp_shape': True,
            'infer': __class__.infer,
            'expand_axis': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        input_node = node.in_nodes()[0]
        output_node = node.out_node()
        if input_node.shape is None:
            return

        assert len(node.in_nodes()) == 1, 'Wrong number of inputs to the layer {}'.format(node.soft_get('name'))

        if not node.has_valid('expand_axis'):
            raise Error('ExpandDims axis is not defined for node {}'.format(node.soft_get('name')))

        expand_axes = node.expand_axis
        if expand_axes is None:
            raise Error('The "expand_axis" attribute is None for node "{}"'.format(node.soft_get('name')))

        if isinstance(expand_axes, int):
            expand_axes = int64_array([expand_axes])
        elif expand_axes.ndim == 0:
            expand_axes = expand_axes.reshape([1])

        # expand_axis is a position where the new axis is placed so expand_dims works for negative axis in a different
        # way not as insert operation
        for expand_axis in expand_axes:
            if expand_axis < 0:
                expand_axis += len(input_node.shape) + 1

        expand_axes = sorted(expand_axes)

        for expand_axis in expand_axes:
            output_node.shape = np.insert(input_node.shape, expand_axis, [1])
        # convert data type of the shape to int64 explicitly
        output_node.shape = output_node.shape.astype(np.int64)
        if input_node.value is not None:
            output_node.value = input_node.value.reshape(output_node.shape)
