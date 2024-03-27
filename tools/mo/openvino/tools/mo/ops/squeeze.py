# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension, shape_delete, is_fully_defined, \
    undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class Squeeze(Op):
    op = 'Squeeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'squeeze_dims': None,
            'reinterp_shape': True,
            'keep_at_least_1d': 0,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        real_squeeze_dims = int64_array([])
        input_shape = node.in_port(0).data.get_shape()
        node_name = node.soft_get('name', node.id)
        if input_shape is None:
            raise Error('Input shape is not defined for node {}'.format(node_name))

        output_shape = input_shape.copy()
        assert len(node.in_nodes()) == 2, 'The Squeeze node {} must have 2 inputs'.format(node_name)

        # TODO remove the following 'if' statement when OV start support 0D tensors
        squeeze_dims = node.in_port(1).data.get_value()
        if squeeze_dims.ndim == 0:
            squeeze_dims = squeeze_dims.reshape([1])

        for dim in squeeze_dims:
            if output_shape[dim] == 1 or output_shape[dim] is dynamic_dimension:
                real_squeeze_dims = np.ma.append(real_squeeze_dims, get_canonical_axis_index(output_shape, dim))
            else:
                raise Error('Trying to squeeze dimension not equal to 1 for node "{}"'.format(node_name))

        # if squeeze_dims empty then all 1s should be removed (tf specification of Squeeze op)
        if squeeze_dims.size == 0:
            for i in range(output_shape.size):
                if output_shape[i] == 1:
                    real_squeeze_dims = np.ma.append(real_squeeze_dims, get_canonical_axis_index(output_shape, i))

        assert is_fully_defined(real_squeeze_dims), 'Squeeze dimension(s) is not defined for op "{}"'.format(node_name)
        output_shape = shape_delete(output_shape, real_squeeze_dims)
        node.out_port(0).data.set_shape(output_shape)

        # make dimensions positive to correctly translate from NHWC to NCHW layout
        if node.in_port(1).get_source().node.op == 'Const':
            node.in_port(1).data.set_value(real_squeeze_dims)

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))

        # the squeeze_dim attribute will be converted to the second input in the end of the Middle phase
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')

    @staticmethod
    def reverse_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()
        squeeze_dims = node.in_port(1).data.get_value()
        if input_shape is None and output_shape is not None and squeeze_dims is not None:
            num_squeeze_dims = 1 if int64_array(squeeze_dims).ndim == 0 else len(squeeze_dims)
            shape = undefined_shape_of_rank(len(output_shape) + num_squeeze_dims)
            node.in_port(0).data.set_shape(shape)
