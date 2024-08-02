# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, compatible_dims
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class CTCGreedyDecoderOp(Op):
    op = 'CTCGreedyDecoder'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',

            'infer': self.infer,
            'reinterp_shape': True,

            'in_ports_count': 2,
            'out_ports_count': 1,

            'ctc_merge_repeated': True
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            ('ctc_merge_repeated', lambda node: bool_to_str(node, 'ctc_merge_repeated'))
        ]

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 2, \
            "Incorrect number of inputs for {} node".format(node_name)

        logits_shape = node.in_port(0).data.get_shape()
        sequence_mask_shape = node.in_port(1).data.get_shape()

        # check shapes of input tensors
        assert len(logits_shape) == 3, \
            'Incorrect rank of logits for {} node'.format(node_name)
        assert len(sequence_mask_shape) == 2, \
            'Incorrect rank of sequence length tensor for {} node'.format(node_name)
        assert compatible_dims(logits_shape[1], sequence_mask_shape[1]), \
            'Batch dimensions of input tensors must be the same for {} node'.format(node_name)
        assert compatible_dims(logits_shape[0], sequence_mask_shape[0]), \
            'Time dimensions of input tensors must be the same for {} node'.format(node_name)

        batch_size = logits_shape[1]
        time_size = logits_shape[0]
        node.out_port(0).data.set_shape([batch_size, time_size, 1, 1])
