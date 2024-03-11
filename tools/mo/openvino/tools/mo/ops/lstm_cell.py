# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins, compatible_dims
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class LSTMCell(Op):
    ''' A single LSTM cell (without a loop).

        3 inputs:
            - [0, required] input data (2D),
            - [1, required] initial hidden state (2D),
            - [2, required] initial cell state (2D),
        
        2 blobs:
            - [3, required] LSTM FC weights
            - [4, required] LSTM FC biases
        
        2 outputs:
            - [required] output data / resulting hidden state (2D)
            - [required] resulting cell state (2D)
    '''
    op = 'LSTMCell'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset4',
            'infer': self.infer,
            'in_ports_count': 5,
            'out_ports_count': 2,
            'wr_input_id': 3,
            'gates_count': 4
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
            ('activations', lambda node: ','.join(node['activations']) if node.has_and_set('activations') else None),
            ('activations_alpha', lambda node: ','.join(map(str, node['activations_alpha']))
                if node.has_and_set('activations_alpha') else None),
            ('activations_beta', lambda node: ','.join(map(str, node['activations_beta']))
                if node.has_and_set('activations_beta') else None),
            'clip',
        ]

    @staticmethod
    def infer(node: Node):
        if node.has_and_set('extra_inputs'):
            assert len(node.in_nodes()) == 8
        else:
            assert len(node.in_nodes()) == 5
        assert len(node.out_nodes()) in [1, 2]

        hidden_shape = node.in_node(1).shape.copy()
        cell_shape = node.in_node(2).shape.copy()

        mark_input_bins(node, start_port=3)
        node.out_node(0).shape = hidden_shape
        if len(node.out_nodes()) == 2:
            node.out_node(1).shape = cell_shape

        hidden_size = hidden_shape[1]

        if node.has_valid('hidden_size'):
            if node.hidden_size != hidden_size:
                raise Error("Input shape {} for hidden size doesn't match pre-defined hidden_size in node {}".format(
                    node.in_node(1).shape, node.soft_get('name')))
        else:
            node['hidden_size'] = hidden_size

        assert cell_shape[1] == hidden_size

        input_shape = node.in_node(0).shape
        assert input_shape is not None
        assert compatible_dims(hidden_shape[0], cell_shape[0]) and \
               compatible_dims(cell_shape[0], input_shape[0]), 'States are not broadcast-able by batch for node {}' \
                                                               ''.format(node.soft_get('name', node.id))
