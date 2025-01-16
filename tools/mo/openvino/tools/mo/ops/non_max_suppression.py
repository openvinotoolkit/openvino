# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, shape_array, dynamic_dimension_value, \
    set_input_shapes, undefined_shape_of_rank
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class NonMaxSuppression(Op):
    op = 'NonMaxSuppression'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset9',
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'output_type': np.int64,
            'box_encoding': 'corner',
            'in_ports_count': 5,
            'sort_result_descending': 1,
            'force_precision_in_ports': {
                2: 'int64'},
            'type_infer': self.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)
        version = self.get_opset()
        if version in ['opset1', 'opset3', 'opset4']:
            self.attrs['out_ports_count'] = 1
        elif version in ['opset5', 'opset9']:
            self.attrs['out_ports_count'] = 3
        else:
            raise Error('Unsupported operation opset version "{}"'.format(version))

    def backend_attrs(self):
        version = self.get_opset()
        if version in ['opset3', 'opset4', 'opset5', 'opset9']:
            return [('sort_result_descending', lambda node: bool_to_str(node, 'sort_result_descending')),
                    'box_encoding',
                    ('output_type', lambda node: np_data_type_to_destination_type(node.output_type))]
        elif version == 'opset1':
            return [('sort_result_descending', lambda node: bool_to_str(node, 'sort_result_descending')),
                    'box_encoding']
        else:
            raise Error('Unsupported operation opset version "{}"'.format(version))

    @staticmethod
    def infer(node: Node):
        num_of_inputs = len(node.in_ports())
        opset = node.get_opset()
        max_num_of_inputs = 6 if opset in ['opset5', 'opset9'] else 5
        input_msg_fmt = 'NonMaxSuppression node {} from {} must have from 2 to {} inputs'
        node_name = node.soft_get('name', node.id)
        inputs_msg = input_msg_fmt.format(node_name, opset, max_num_of_inputs)
        assert 2 <= num_of_inputs <= max_num_of_inputs, inputs_msg

        boxes_shape = node.in_port(0).data.get_shape()
        assert boxes_shape is not None, 'The shape of tensor with boxes is not defined'
        scores_shape = node.in_port(1).data.get_shape()
        assert scores_shape is not None, 'The shape of tensor with scores is not defined'
        assert len(boxes_shape) == 3, 'Length of tensors with boxes must be equal to 3'
        assert len(scores_shape) == 3, 'Length of tensors with scores must be equal to 3'

        # According to the specification of the operation NonMaxSuppression,
        # the input 'max_output_boxes_per_class' (port 2) is optional, with default value 0.
        if num_of_inputs >= 3:
            max_output_boxes_per_class = node.in_port(2).data.get_value()
        else:
            max_output_boxes_per_class = 0

        if not max_output_boxes_per_class:
            log.info('Set default "max_output_boxes_per_class" for node {} to number of boxes'.format(node.name))
            max_output_boxes_per_class = boxes_shape[1]

        # convert the np.array value to a scalar to avoid issue with ragged numpy array generation in the shape
        # calculation formulas below
        if isinstance(max_output_boxes_per_class, np.ndarray):
            max_output_boxes_per_class = max_output_boxes_per_class.item()

        num_classes = scores_shape[1]
        num_input_boxes = boxes_shape[1]
        assert scores_shape[2] is dynamic_dimension or scores_shape[2] == num_input_boxes or scores_shape[2] is None \
               or num_input_boxes is None, 'Number of boxes mismatch for operation {}'.format(node_name)

        if node.get_opset() in ['opset4', 'opset5', 'opset9']:
            max_number_of_boxes = min(num_input_boxes, max_output_boxes_per_class) * boxes_shape[0] * num_classes
        else:
            max_number_of_boxes = min(num_input_boxes, boxes_shape[0] * max_output_boxes_per_class * num_classes)
        node.out_port(0).data.set_shape(shape_array([max_number_of_boxes, 3]))

        if opset in ['opset5', 'opset9']:
            node.out_port(0).data.set_shape(shape_array([dynamic_dimension_value, 3]))
            num_of_outputs = len([port for port in node.out_ports().values() if not port.disconnected()])
            if num_of_outputs >= 2 and node.has_port('out', 1):
                node.out_port(1).data.set_shape(shape_array([dynamic_dimension_value, 3]))
            if num_of_outputs >= 3 and node.has_port('out', 2):
                node.out_port(2).data.set_shape(shape_array([1]))

    @staticmethod
    def type_infer(node):
        opset = node.get_opset()
        if opset in ['opset5', 'opset9']:
            node.out_port(0).set_data_type(node.output_type)
            if node.has_port('out', 1):
                node.out_port(1).set_data_type(np.float32)
            if node.has_port('out', 2):
                node.out_port(2).set_data_type(np.int64)
        elif opset in ['opset3', 'opset4']:
            node.out_port(0).set_data_type(node.output_type)
        else:
            node.out_port(0).set_data_type(np.int64)

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node, undefined_shape_of_rank(3), undefined_shape_of_rank(3))
