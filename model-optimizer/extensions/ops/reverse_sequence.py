# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.op import Op


class ReverseSequence(Op):
    op = 'ReverseSequence'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'version': 'opset1',
            'seq_axis': None,
            'batch_axis': 0,
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'seq_axis', 'batch_axis',
        ]
    
    @staticmethod
    def infer(node):
        input_data_shape = node.in_node(0).shape
        assert input_data_shape is not None
        assert node.has_valid('seq_axis')
        assert node.has_valid('batch_axis')

        assert len(node.out_nodes()) == 1
        node.out_node().shape = input_data_shape.copy()
