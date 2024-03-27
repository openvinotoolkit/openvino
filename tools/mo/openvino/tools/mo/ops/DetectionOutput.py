# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, compatible_dims, \
    undefined_shape_of_rank, set_input_shapes
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class DetectionOutput(Op):
    op = 'DetectionOutput'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset8',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
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
        supported_attrs = [
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
            ('share_location', lambda node: bool_to_str(node, 'share_location')),
            'top_k',
            ('variance_encoded_in_target', lambda node: bool_to_str(node, 'variance_encoded_in_target')),
            'objectness_score',
        ]
        opset = self.get_opset()
        if opset == 'opset1':
            supported_attrs += ['num_classes']
        return supported_attrs

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(np.float32)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        loc_shape = node.in_port(0).data.get_shape()
        conf_shape = node.in_port(1).data.get_shape()
        prior_boxes_shape = node.in_port(2).data.get_shape()

        if loc_shape is None or conf_shape is None or prior_boxes_shape is None:
            raise Error('Shapes for the Detection Output node "{}" are not defined'.format(node_name))

        prior_size = 4
        if node.has('normalized') and not node.normalized:
            prior_size = 5

        if is_fully_defined(prior_boxes_shape[-1]) and prior_boxes_shape[-1] % prior_size != 0:
            raise Error('Amount of confidences "{}" is not divisible by {} for node "{}"'
                        ''.format(prior_boxes_shape[-1], prior_size, node_name))

        num_priors = prior_boxes_shape[-1] // prior_size
        if not node.has_valid('keep_top_k') or node.keep_top_k == -1:
            node['keep_top_k'] = num_priors

        num_classes = conf_shape[-1] // num_priors
        num_loc_classes = num_classes
        if node.has_and_set('share_location') and node.share_location:
            num_loc_classes = 1

        if not compatible_dims(num_priors * num_loc_classes * 4, loc_shape[-1]):
            raise Error('Locations and prior boxes shapes mismatch: "{}" vs "{}" for node "{}"'
                        ''.format(loc_shape, prior_boxes_shape, node_name))

        if not node.variance_encoded_in_target and not compatible_dims(prior_boxes_shape[-2], 2):
            raise Error('The "-2" dimension of the prior boxes must be 2 but it is "{}" for node "{}".'
                        ''.format(prior_boxes_shape[-2], node_name))

        if is_fully_defined(conf_shape[-1]) and is_fully_defined(num_priors) and conf_shape[-1] % num_priors != 0:
            raise Error('Amount of confidences "{}" is not divisible by amount of priors "{}" for node "{}".'
                        ''.format(conf_shape[-1], num_priors, node_name))

        node.out_port(0).data.set_shape([1, 1, conf_shape[0] * node.keep_top_k, 7])

        # the line below is needed for the TF framework so the MO will not change the layout
        node.graph.node[node.out_node(0).id]['nchw_layout'] = True

    @staticmethod
    def reverse_infer(node):
        num_in_ports = len(node.in_ports())
        assert num_in_ports in [3, 6], 'incorrect number of input ports for DetectionOutput node {}'.format(node.soft_get('name', node.id))
        if num_in_ports == 3:
            set_input_shapes(node,
                             undefined_shape_of_rank(2),
                             undefined_shape_of_rank(2),
                             undefined_shape_of_rank(3))
        elif num_in_ports == 6:
            set_input_shapes(node,
                             undefined_shape_of_rank(2),
                             undefined_shape_of_rank(2),
                             undefined_shape_of_rank(3),
                             undefined_shape_of_rank(2),
                             undefined_shape_of_rank(2))
