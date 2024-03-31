# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op, PermuteAttrs


class ReverseSequence(Op):
    op = 'ReverseSequence'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'version': 'opset1',
            'seq_axis': None,
            'batch_axis': 0,
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'seq_axis', 'batch_axis',
        ]
    
    @staticmethod
    def infer(node):
        input_data_shape = node.in_port(0).data.get_shape()
        assert input_data_shape is not None
        assert node.has_valid('seq_axis')
        assert node.has_valid('batch_axis')

        assert len(node.out_nodes()) == 1
        node.out_port(0).data.set_shape(input_data_shape)

        PermuteAttrs.create_permute_attrs(node, attrs=[('seq_axis', 'input:0')])
        PermuteAttrs.create_permute_attrs(node, attrs=[('batch_axis', 'input:0')])
