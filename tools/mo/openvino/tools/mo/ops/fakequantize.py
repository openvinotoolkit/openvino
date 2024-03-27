# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


def broadcastable(broadcast_from, broadcast_to):
    """Check if shape broadcast_from can be broadcasted to broadcast_to"""
    broadcast_to = int64_array(broadcast_to)
    broadcast_from = int64_array(broadcast_from)
    if broadcast_from.size > broadcast_to.size:
        return False
    broadcast_from = np.concatenate(
        (int64_array([1] * (broadcast_to.size - broadcast_from.size)), broadcast_from))
    return np.all(np.logical_or(broadcast_from == 1, broadcast_from == broadcast_to))


def round_half_up(n):
    return np.floor(n + 0.5)


class FakeQuantize(Op):
    op = 'FakeQuantize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'levels': None,
            'is_eltwise': True,
            'infer': self.infer,
            'in_ports_count': 5,
            'out_ports_count': 1,
            'auto_broadcast': 'numpy'
        }
        super().__init__(graph, mandatory_props, attrs)
        if self.attrs['levels'] is None:
            raise Error("FakeQuantize operation has no levels parameter")

    def supported_attrs(self):
        return [
            'levels',
            'auto_broadcast'
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 5
        assert len(node.out_nodes()) == 1
        inputs = [node.in_node(i) for i in range(5)]
        x, input_low, input_high, output_low, output_high = inputs
        assert x.has_valid('shape')
        # TODO Check all inputs[1..4] shapes are broadcastable to inputs[0] shape
        assert all([broadcastable(inputs[i].shape, inputs[0].shape) for i in range(1, 5)]), \
            "Not all shapes from FakeQuantize inputs can be broadcasted to input[0] for node {}".format(
                node.soft_get('name'))
        node.out_node().shape = x.shape.copy()

        if all([node.in_node(i).has_valid('value') for i in range(5)]):
            x, input_low, input_high, output_low, output_high = \
                [float32_array(np.broadcast_to(node.value, x.value.shape)) for node in inputs]

            assert node.has_valid('levels')
            assert isinstance(node.levels, int)

            underflow_mask = x <= input_low
            overflow_mask = x > input_high
            # pylint: disable=assignment-from-no-return
            middle_mask = np.logical_not(np.logical_or(underflow_mask, overflow_mask))

            def middle_part(x, input_low, input_high, output_low, output_high):
                return round_half_up((x - input_low) / (input_high - input_low) * (node.levels - 1)) / \
                    (node.levels - 1) * (output_high - output_low) + output_low

            output = np.zeros_like(x)
            # pylint: disable=unsupported-assignment-operation
            output[middle_mask] = middle_part(
                x[middle_mask],
                input_low[middle_mask],
                input_high[middle_mask],
                output_low[middle_mask],
                output_high[middle_mask],
            )

            # pylint: disable=unsupported-assignment-operation
            output[overflow_mask] = output_high[overflow_mask]
            # pylint: disable=unsupported-assignment-operation
            output[underflow_mask] = output_low[underflow_mask]

            if not node.has_and_set('stop_value_propagation'):
                node.out_node().value = output
