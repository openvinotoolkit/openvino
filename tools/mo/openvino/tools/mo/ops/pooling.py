# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import tf_window_op_pad_infer, int64_array, shape_array, \
    dynamic_dimension_value, dynamic_dimension, undefined_shape_of_rank
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.front.onnx.extractors.utils import get_backend_pad
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.utils.error import Error


poolings_map = {
    'max': {'version': 'opset8', 'out_ports_count': 2},
    'avg': {'version': 'opset1', 'out_ports_count': 1}
}


class PoolingV2(Op):
    """
    TensorFlow MaxPoolV2 and AvgPoolV2 operations expect windows_size and strides values from inputs not from
    attributes. This internal operation is introduced to handle that. Only constant windows_size and strides
    values are supported. Eventually will be replaced with the standard pooling operations from the opset.
    """
    op = 'PoolingV2'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'version': None,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        assert (len(node.in_nodes()) == 3), 'MaxPoolV2 node {} from must have only 3 inputs: input, window size, and ' \
                                            'strides but instead got {} inputs'.format(node.soft_get('name', node.id),
                                                                                       len(node.in_nodes()))
        node['window'] = node.in_port(1).data.get_value()
        node['stride'] = node.in_port(2).data.get_value()

        if node['window'] is None:
            raise Error('The non-constant window size for MaxPoolV2 node {} is not supported'
                        ''.format(node.soft_get('name', node.id)))
        if node['stride'] is None:
            raise Error('The non-constant strides for MaxPoolV2 node {} is not supported'
                        ''.format(node.soft_get('name', node.id)))

        Pooling.pool_infer(node)

    @staticmethod
    def reverse_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        window_shape = node.in_port(1).data.get_shape()
        # use the value of the 'window' input to determine input tensor rank
        if input_shape is None and window_shape is not None:
            node.in_port(0).data.set_shape(undefined_shape_of_rank(window_shape[0]))


class Pooling(Op):
    op = 'Pooling'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': poolings_map[attrs.get('pool_method')]['version'],
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'in_ports_count': 1,
            'out_ports_count': 1 if attrs.get('version') == 'opset1' else
            poolings_map[attrs.get('pool_method')]['out_ports_count']
        }, attrs)

    def backend_attrs(self):
        backend_attrs_list = [
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('kernel', lambda node: ','.join(map(str, node['window'][node.spatial_dims]))),

            ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0)))),
            ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1)))),

            ('exclude-pad', lambda node: bool_to_str(node, 'exclude_pad')),

            'rounding_type',
            ('auto_pad', lambda node: node.auto_pad if node.has_valid('auto_pad') else 'explicit')
        ]

        if self.attrs.get('pool_method') == 'avg':
            return backend_attrs_list
        else:
            return backend_attrs_list + [
                ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims]))),
                'axis',
                ('index_element_type', lambda node: np_data_type_to_destination_type(node.index_element_type))
            ]

    @staticmethod
    def infer(node: Node):
        assert (len(node.in_nodes()) == 1), 'MaxPool node {} from must have only one input but instead got ' \
                                            '{} inputs'.format(node.soft_get('name', node.id), len(node.in_nodes()))

        Pooling.pool_infer(node)

    @staticmethod
    def pool_infer(node: Node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        if not node.has_valid('spatial_dims'):
            node['spatial_dims'] = np.delete([x for x in range(len(input_shape))],
                                             [node.batch_dims[0], node.channel_dims[0]])

        input_spatial_shape = input_shape[node.spatial_dims]

        # Setting default pad and stride attrs in case if None specified
        if not node.has_valid('pad'):
            node['pad'] = int64_array([[0, 0] for x in range(len(input_shape))])
        if not node.has_valid('pad_spatial_shape'):
            node['pad_spatial_shape'] = node.pad[node.spatial_dims]

        if not node.has_valid('stride'):
            node['stride'] = int64_array([1 for x in range(len(input_shape))])

        if node.has_and_set('global_pool'):
            node['window'] = np.zeros(len(input_shape), dtype=np.int64)
            node.window[node.spatial_dims] = input_spatial_shape

        if not node.has_valid('dilation'):
            node['dilation'] = np.ones(len(input_shape), dtype=np.float32)

        if not node.has_valid('axis'):
            node['axis'] = 0

        if not node.has_valid('index_element_type'):
            node['index_element_type'] = np.int64

        window_spatial_shape = node.window[node.spatial_dims]
        stride_spatial = node.stride[node.spatial_dims]
        dilation_spatial = node.dilation[node.spatial_dims]
        assert any(stride_spatial), 'Stride can not be zero in node {}'.format(node.id)

        if node.has_valid('auto_pad') and node.auto_pad != 'explicit':
            node.pad_spatial_shape, node.output_spatial_shape = tf_window_op_pad_infer(input=input_spatial_shape,
                                                                                       window=window_spatial_shape,
                                                                                       stride=stride_spatial,
                                                                                       auto_pad=node.auto_pad,
                                                                                       dilation=dilation_spatial)
            pad = np.zeros((len(input_shape), 2), dtype=np.int64)
            pad[node.spatial_dims] = node.pad_spatial_shape
            node.pad = pad
        else:

            pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)

            rounding = np.floor
            if node.soft_get('pooling_convention') == 'full' or node.soft_get('rounding_type') == 'ceil':
                rounding = np.ceil

            padded_spatial_shape = input_spatial_shape + pad_spatial_shape - ((window_spatial_shape - 1) *
                                                                              dilation_spatial + 1)
            if np.any(padded_spatial_shape < 0):
                raise Error("Data after padding has dimension less than window size. " +
                            "Possible reason of error is incorrectly specified model input shape(s).")

            output_spatial_shape = shape_array([dynamic_dimension_value for _ in range(len(padded_spatial_shape))])
            for idx in range(len(padded_spatial_shape)):
                if padded_spatial_shape[idx] is not dynamic_dimension and stride_spatial[idx] is not dynamic_dimension:
                    output_spatial_shape[idx] = int(rounding(padded_spatial_shape[idx] / stride_spatial[idx])) + 1

            original_pads = mo_array([i[1] for i in node.pad_spatial_shape])

            for i in range(len(input_spatial_shape)):
                if original_pads[i] and (output_spatial_shape[i] - 1) * stride_spatial[i] >= \
                        input_spatial_shape[i] + original_pads[i]:
                    output_spatial_shape[i] -= 1

            node['output_spatial_shape'] = output_spatial_shape

        output_shape = input_shape.copy()
        output_shape[node.spatial_dims] = node.output_spatial_shape
        node.out_port(0).data.set_shape(output_shape)

        if len(node.out_ports()) == 2 and not node.out_port(1).disconnected():
            node.out_port(1).data.set_shape(output_shape)

        if node.has_and_set('pool_method') and node['pool_method'] == 'max':
            node['remove_values_output'] = True

        # Add permute_attrs
        PermuteAttrs.create_permute_attrs(node, attrs=[('pad', 'input:0'),
                                                       ('stride', 'input:0'),
                                                       ('window', 'input:0'),
                                                       ('spatial_dims', 'input:0'),
                                                       ('dilation', 'input:0')])

    @staticmethod
    def reverse_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        window = node.soft_get('window', None)
        if input_shape is None and window is not None:
            node.in_port(0).data.set_shape(undefined_shape_of_rank(len(window)))
