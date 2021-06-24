# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.multi_box_detection import multi_box_detection_infer
from mo.front.extractor import bool_to_str
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
            'normalized': True,
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
            ('clip_after_nms', lambda node: bool_to_str(node, 'clip_after_nms')),
            ('clip_before_nms', lambda node: bool_to_str(node, 'clip_before_nms')),
            'code_type',
            'confidence_threshold',
            ('decrease_label_id', lambda node: bool_to_str(node, 'decrease_label_id')),
            'input_height',
            'input_width',
            'keep_top_k',
            'nms_threshold',
            ('normalized', lambda node: bool_to_str(node, 'normalized')),
            'num_classes',
            ('share_location', lambda node: bool_to_str(node, 'share_location')),
            'top_k',
            ('variance_encoded_in_target', lambda node: bool_to_str(node, 'variance_encoded_in_target')),
            'objectness_score',
        ]

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(np.float32)
