# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, is_fully_defined, shape_insert, undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class Unsqueeze(Op):
    """
    The operation that inserts dimensions of size one into specific positions of the input layer. The dimensions are
    specified in the second input.
    """
    op = 'Unsqueeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'unsqueeze_dims': None,
            'reinterp_shape': True,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
        }, attrs)

    @staticmethod
    def infer(node):
        if len(node.in_nodes()) <= 1:
            raise Error('There is no input with unsqueeze dims for the node {}'.format(node.soft_get('name')))
        unsqueeze_dims = node.in_port(1).data.get_value()
        if unsqueeze_dims is None:
            raise Error('The dimensions to unsqueeze are not defined for the node {}'.format(node.soft_get('name')))
        unsqueeze_dims = int64_array(unsqueeze_dims)

        input_value = node.in_port(0).data.get_value()
        input_shape = node.in_port(0).data.get_shape()

        # TODO remove the following line when the OpenVINO plugins support 0D tensors
        if unsqueeze_dims.ndim == 0:
            unsqueeze_dims = int64_array([unsqueeze_dims.item()])

        # make dimensions positive to correctly translate from NHWC to NCHW layout
        unsqueeze_dims = int64_array([dim + len(node.in_port(0).data.get_shape()) + 1 if dim < 0 else dim
                                      for dim in unsqueeze_dims])
        if node.in_port(1).get_source().node.op == 'Const':
            node.in_port(1).data.set_value(unsqueeze_dims)

        output_shape = input_shape.copy()
        for dim in unsqueeze_dims:
            output_shape = shape_insert(output_shape, dim, 1)

        if input_value is not None and is_fully_defined(output_shape):
            node.out_port(0).data.set_value(input_value.reshape(output_shape))
        else:
            node.out_port(0).data.set_shape(output_shape)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

    @staticmethod
    def reverse_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()
        unsqueeze_dims = node.in_port(1).data.get_value()
        if input_shape is None and output_shape is not None and unsqueeze_dims is not None:
            num_unsqueeze_dims = 1 if int64_array(unsqueeze_dims).ndim == 0 else len(unsqueeze_dims)
            shape = undefined_shape_of_rank(len(output_shape) - num_unsqueeze_dims)
            node.in_port(0).data.set_shape(shape)
