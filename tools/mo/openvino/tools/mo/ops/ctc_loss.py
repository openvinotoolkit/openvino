# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, compatible_dims
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class CTCLoss(Op):
    op = 'CTCLoss'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset4',

            'type_infer': self.type_infer,
            'infer': self.infer,

            'in_ports_count': 5,
            'out_ports_count': 1,

            'preprocess_collapse_repeated': False,
            'ctc_merge_repeated': True,
            'unique': False
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [('preprocess_collapse_repeated', lambda node: bool_to_str(node, 'preprocess_collapse_repeated')),
                ('ctc_merge_repeated', lambda node: bool_to_str(node, 'ctc_merge_repeated')),
                ('unique', lambda node: bool_to_str(node, 'unique'))]

    @staticmethod
    def type_infer(node):
        logits_type = node.in_port(0).get_data_type()
        logit_length_type = node.in_port(1).get_data_type()
        labels_type = node.in_port(2).get_data_type()
        label_length_type = node.in_port(3).get_data_type()
        blank_index_type = labels_type
        if not node.in_port(4).disconnected():
            blank_index_type = node.in_port(4).get_data_type()

        assert logit_length_type == label_length_type and logit_length_type in [np.int64, np.int32], \
            'Inputs with logits and labels lengths for node {} must be the same and int32 or int64, {} and {} found'.format(
                node.soft_get('name'), logit_length_type, label_length_type)
        assert labels_type == blank_index_type and labels_type in [np.int64, np.int32], \
            'Inputs with labels and blank index for node {} must be the same and int32 or int64, {} and {} found'.format(
                node.soft_get('name'), labels_type, blank_index_type)

        node.out_port(0).set_data_type(logits_type)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) in [4, 5], \
            "Incorrect number of inputs for {} node".format(node_name)

        logits_shape = node.in_port(0).data.get_shape()
        logit_length_shape = node.in_port(1).data.get_shape()
        labels_shape = node.in_port(2).data.get_shape()
        label_length_shape = node.in_port(3).data.get_shape()
        blank_index_shape = int64_array([])
        if len(node.in_nodes()) == 5:
            blank_index_shape = node.in_port(4).data.get_shape()

        # check shapes of input tensors
        assert len(logits_shape) == 3 and len(logit_length_shape) == 1 and len(labels_shape) == 2\
            and len(label_length_shape) == 1 and len(blank_index_shape) == 0, \
            'Incorrect rank of some input tensor for {} node'.format(node_name)
        assert compatible_dims(logits_shape[0], logit_length_shape[0]) and \
               compatible_dims(logits_shape[0], labels_shape[0]) and \
               compatible_dims(logits_shape[0], label_length_shape[0]), \
            'Batch dimensions of input tensors must be the same for {} node'.format(node_name)
        assert compatible_dims(logits_shape[1], labels_shape[1]), \
            'Time dimensions of input tensors must be the same for {} node'.format(node_name)

        batch_size = logits_shape[0]
        node.out_port(0).data.set_shape([batch_size])
