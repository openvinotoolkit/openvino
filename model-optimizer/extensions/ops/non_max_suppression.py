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
import logging as log

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class NonMaxSuppression(Op):
    op = 'NonMaxSuppression'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'center_point_box': 0,
            'box_encoding': 'corner',
            'in_ports_count': 5,
            'out_ports_count': 1,
            'sort_result_descending': 1,
            'force_precision_in_ports': {
                2: 'int64' if graph.graph['cmd_params'].generate_experimental_IR_V10 else 'int32'},
            'type_infer': self.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        if self.ir_version < 10:
            return [
                'center_point_box',
            ]
        else:
            return [
                'sort_result_descending',
                'box_encoding'
            ]


    @staticmethod
    def infer(node: Node):
        boxes_shape = node.in_port(0).data.get_shape()
        assert boxes_shape is not None, 'The shape of tensor with boxes is not defined'
        scores_shape = node.in_port(1).data.get_shape()
        assert scores_shape is not None, 'The shape of tensor with scores is not defined'
        assert len(boxes_shape) == 3, 'Length of tensors with boxes must be equal to 3'
        assert len(scores_shape) == 3, 'Length of tensors with scores must be equal to 3'

        max_output_boxes_per_class = node.in_port(2).data.get_value()
        if max_output_boxes_per_class is None:
            log.info('Set default "max_output_boxes_per_class" for node {} to number of boxes'.format(node.name))
            max_output_boxes_per_class = boxes_shape[1]

        num_classes = scores_shape[1]
        num_input_boxes = boxes_shape[1]
        assert scores_shape[2] == num_input_boxes, 'Number of boxes mismatch'

        max_number_of_boxes = min(num_input_boxes, boxes_shape[0] * max_output_boxes_per_class * num_classes)
        node.out_port(0).data.set_shape(int64_array([max_number_of_boxes, 3]))

    @staticmethod
    def type_infer(node):
        node.out_port(0).set_data_type(np.int64 if node.graph.graph['cmd_params'].generate_experimental_IR_V10 else
                                       np.int32)
