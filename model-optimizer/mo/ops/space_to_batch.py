"""
 Copyright (C) 2018-2020 Intel Corporation

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
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class SpaceToBatch(Op):
    op = 'SpaceToBatch'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'version': 'opset2',
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node):
        """
        https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch
        """
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        if len(node.in_nodes()) != 4:
            return

        block_size = node.in_port(1).data.get_value()
        pads_begin = node.in_port(2).data.get_value()
        pads_end = node.in_port(3).data.get_value()
        if block_size is None or pads_begin is None or pads_end is None:
            return

        pads = pads_begin + input_shape + pads_end

        node.out_node().shape = int64_array([input_shape[0] * np.prod(block_size),
                                             *[int(x) for x in (pads[1:] / block_size[1:])]])

        # block_shape, pads_begin, pads_end should be permuted during the NHWC->NCHW layout change
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'shape')


class BatchToSpace(Op):
    op = 'BatchToSpace'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'op': self.op,
            'type': self.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'version': 'opset2',
            'infer': __class__.infer
        }, attrs)

    @staticmethod
    def infer(node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        if len(node.in_nodes()) != 4:
            return

        block_size = node.in_port(1).data.get_value()
        crops_begin = node.in_port(2).data.get_value()
        crops_end = node.in_port(3).data.get_value()
        if block_size is None or crops_begin is None or crops_end is None:
            return

        pads = block_size * input_shape

        sizes = pads[1:] - crops_begin[1:] - crops_end[1:]
        batch = int(input_shape[0] / (np.prod(block_size)))

        node.out_node().shape = int64_array([batch, *sizes])

        # block_shape, crops_begin, crops_end values should be permuted during the NHWC->NCHW layout change
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'shape')
