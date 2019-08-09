"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error


def broadcastable(broadcast_from, broadcast_to):
    """Check if shape broadcast_from can be broadcasted to broadcast_to"""
    broadcast_to = np.array(broadcast_to, dtype=np.int64)
    broadcast_from = np.array(broadcast_from, dtype=np.int64)
    if broadcast_from.size > broadcast_to.size:
        return False
    broadcast_from = np.concatenate(
        (np.array([1] * (broadcast_to.size - broadcast_from.size), dtype=np.int64), broadcast_from))
    return np.all(np.logical_or(broadcast_from == 1, broadcast_from == broadcast_to))


class FakeQuantize(Op):
    op = 'FakeQuantize'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'levels': None,
            # flag to switch between dumping FakeQuantize as statistics and keeping it as layer in IR
            'keep_in_IR': None,
            'infer': __class__.infer,
            'in_ports_count': 5,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)
        if self.attrs['levels'] is None:
            raise Error("FakeQuantize operation has no levels parameter")
        # TODO remove following lines after FakeQuantize supported for int8 workflow
        self.attrs['keep_in_IR'] = self.attrs['levels'] == 2 or graph.graph['cmd_params'].keep_quantize_ops_in_IR

    def supported_attrs(self):
        return [
            'levels',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 5
        assert len(node.out_nodes()) == 1
        inputs = [node.in_node(i) for i in range(5)]
        x, input_low, input_high, output_low, output_high = inputs
        assert x.has_valid('shape')
        # TODO Check all input[1..4] shapes are broadcastable to intput[0] shape
        assert all([broadcastable(inputs[i].shape, inputs[0].shape) for i in range(1, 5)]), \
            "Not all shapes from FakeQuantize inputs can be broadcasted to input[0] for node {}".format(
                node.soft_get('name'))
        node.out_node().shape = x.shape.copy()

        if all([node.in_node(i).has_valid('value') for i in range(5)]):
            x, input_low, input_high, output_low, output_high = \
                [np.array(np.broadcast_to(node.value, x.value.shape)) for node in inputs]

            assert node.has_valid('levels')
            assert isinstance(node.levels, int)

            underflow_mask = x <= input_low
            overflow_mask = x > input_high
            # pylint: disable=assignment-from-no-return
            middle_mask = np.logical_not(np.logical_or(underflow_mask, overflow_mask))

            def middle_part(x, input_low, input_high, output_low, output_high):
                return np.round(
                    (x - input_low) / (input_high - input_low) * (node.levels - 1)
                ) / (node.levels - 1) * (output_high - output_low) + output_low

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
