# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, dynamic_dimension
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op


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
            'infer': self.infer,
        }, attrs)

    @staticmethod
    def infer(node):
        """
        https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch
        """
        input_shape = node.in_port(0).data.get_shape()
        node_name = node.soft_get('name', node.id)
        assert len(node.in_nodes()) == 4, 'Some inputs are not connected for the operation SpaceToBatch with name {}' \
                                          ''.format(node_name)

        block_size = node.in_port(1).data.get_value()
        pads_begin = node.in_port(2).data.get_value()
        pads_end = node.in_port(3).data.get_value()
        assert block_size is not None and pads_begin is not None and pads_end is not None,\
            'Some inputs are not defined for SpaceToBatch operation with name {}'.format(node_name)

        pads = pads_begin + input_shape + pads_end

        if is_fully_defined(block_size):
            block_elements_count = np.prod(block_size)
        else:
            block_elements_count = dynamic_dimension
        node.out_port(0).data.set_shape([input_shape[0] * block_elements_count,
                                         *[x for x in (pads[1:] // block_size[1:])]])

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
            'infer': self.infer
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
        if is_fully_defined(block_size):
            block_elements_count = np.prod(block_size)
        else:
            block_elements_count = dynamic_dimension
        batch = input_shape[0] // block_elements_count

        node.out_port(0).data.set_shape([batch, *sizes])

        # block_shape, crops_begin, crops_end values should be permuted during the NHWC->NCHW layout change
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'shape')
