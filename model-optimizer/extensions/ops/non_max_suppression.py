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
import logging as log

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op
from mo.utils.error import Error


class NonMaxSuppression(Op):
    op = 'NonMaxSuppression'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset5',
            'infer': self.infer,
            'output_type': np.int64,
            'center_point_box': 0,
            'box_encoding': 'corner',
            'sort_result_descending': 1,
            'out_ports_count': 3,
            'force_precision_in_ports': {
                2: 'int64'},
            'type_infer': self.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version in ['opset3', 'opset4', 'opset5']:
            return ['sort_result_descending', 'box_encoding',
                    ('output_type', lambda node: np_data_type_to_destination_type(node.output_type))]
        elif version == 'opset1':
            return ['sort_result_descending', 'box_encoding']
        else:
            raise Error('Unsupported operation opset version "{}"'.format(version))

    @staticmethod
    def infer(node: Node):
        num_of_inputs = len(node.in_ports())
        assert 2 <= num_of_inputs <= 6, \
            "NonMaxSuppression node {} must have from 2 to 6 inputs".format(node.soft_get(node.name, node.id))

        boxes_shape = node.in_port(0).data.get_shape()
        assert boxes_shape is not None, 'The shape of tensor with boxes is not defined'
        scores_shape = node.in_port(1).data.get_shape()
        assert scores_shape is not None, 'The shape of tensor with scores is not defined'
        assert len(boxes_shape) == 3, 'Length of tensors with boxes must be equal to 3'
        assert len(scores_shape) == 3, 'Length of tensors with scores must be equal to 3'

        if num_of_inputs >= 3:
            max_output_boxes_per_class = node.in_port(2).data.get_value()
        else:
            max_output_boxes_per_class = 0

        if not max_output_boxes_per_class:
            log.info('Set default "max_output_boxes_per_class" for node {} to number of boxes'.format(node.name))
            max_output_boxes_per_class = boxes_shape[1]

        num_classes = scores_shape[1]
        num_input_boxes = boxes_shape[1]
        assert scores_shape[2] == num_input_boxes, 'Number of boxes mismatch'

        if node.get_opset() in ['opset4', 'opset5']:
            max_number_of_boxes = min(num_input_boxes, max_output_boxes_per_class) * boxes_shape[0] * num_classes
        else:
            max_number_of_boxes = min(num_input_boxes, boxes_shape[0] * max_output_boxes_per_class * num_classes)
        node.out_port(0).data.set_shape(int64_array([max_number_of_boxes, 3]))

        num_of_outputs = len([port for port in node.out_ports().values() if not port.disconnected()])
        if num_of_outputs >= 2:
            node.out_port(1).data.set_shape(int64_array([max_number_of_boxes, 3]))
        if num_of_outputs >= 3:
            node.out_port(2).data.set_shape(np.array(1).astype(np.int64))

    @staticmethod
    def type_infer(node):
        opset = node.get_opset()
        if opset == 'opset5':
            node.out_port(0).set_data_type(node.output_type)
            num_of_outputs = len(node.out_ports())
            if num_of_outputs >= 2:
                node.out_port(1).set_data_type(np.int64)
            if num_of_outputs >= 3:
                node.out_port(2).set_data_type(np.int64)
        elif node.get_opset() in ['opset3', 'opset4']:
            node.out_port(0).set_data_type(node.output_type)
        else:
            node.out_port(0).set_data_type(np.int64)
