"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer
from mo.front.extractor import get_boolean_attr
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class DetectionOutput(Op):
    op = 'DetectionOutput'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': multi_box_detection_infer,
            'input_width': 1,
            'input_height': 1,
            'normalized': True,     # In specification we have default value `False`, but it breaks shape inference
            'share_location': True,
            'clip_after_nms': False,
            'clip_before_nms': False,
            'decrease_label_id': False,
            'variance_encoded_in_target': False,
            'type_infer': self.type_infer,
        }, attrs)

    def supported_attrs(self):
        return [
            'background_label_id',
            ('clip_after_nms', lambda node: get_boolean_attr(node, 'clip_after_nms')),
            ('clip_before_nms', lambda node: get_boolean_attr(node, 'clip_before_nms')),
            'code_type',
            'confidence_threshold',
            ('decrease_label_id', lambda node: get_boolean_attr(node, 'decrease_label_id')),
            'input_height',
            'input_width',
            'keep_top_k',
            'nms_threshold',
            ('normalized', lambda node: get_boolean_attr(node, 'normalized')),
            'num_classes',
            ('share_location', lambda node: get_boolean_attr(node, 'share_location')),
            'top_k',
            ('variance_encoded_in_target', lambda node: get_boolean_attr(node, 'variance_encoded_in_target')),
            'objectness_score',
        ]

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(np.float32)
