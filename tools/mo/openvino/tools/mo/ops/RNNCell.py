# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class RNNCell(Op):
    """ A single RNN cell (without a loop).

        2 inputs:
            - [0, required] input data (2D),
            - [1, required] initial hidden state (2D),

        2 blobs:
            - [2, required] cell FC weights
            - [3, required] cell FC biases

        1 outputs:
            - [required] output data / resulting hidden state (2D)
    """
    op = 'RNNCell'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
            'version': 'opset3',
            'wr_input_id': 2,
            'gates_count': 1
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'activations',
            'activation_alpha',
            'activation_beta',
            'clip',
        ]

    def backend_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            ('activations', lambda node: ','.join(node.activations) if node.activations is not None else None),
            'activation_alpha',
            'activation_beta',
            'clip',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.out_nodes()) in [1, 2]

        hidden_shape = node.in_node(1).shape.copy()

        mark_input_bins(node, start_port=2)
        node.out_node(0).shape = hidden_shape

        hidden_size = hidden_shape[1]
        if node.has_valid('hidden_size'):
            if node.hidden_size != hidden_size:
                raise Error("Input shape {} for hidden size doesn't match pre-defined hidden_size in node {}".format(
                    node.in_node(1).shape, node.soft_get('name')))
        else:
            node['hidden_size'] = hidden_size
