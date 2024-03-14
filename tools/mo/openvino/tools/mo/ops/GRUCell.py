# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins
from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class GRUCell(Op):
    """ A single GRU cell (without a loop).

        2 inputs:
            - [0, required] input data (2D),
            - [1, required] initial hidden state (2D),

        2 blobs:
            - [2, required] cell FC weights
            - [3, required] cell FC biases

        1 outputs:
            - [required] output data / resulting hidden state (2D)
    """
    op = 'GRUCell'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
            'version': 'opset3',
            'wr_input_id': 2,
            'gates_count': 3,
            'linear_before_reset': False,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'activations',
            'activation_alpha',
            'activation_beta',
            'clip',
            'linear_before_reset',
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
            ('linear_before_reset', lambda node: bool_to_str(node, 'linear_before_reset')),
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.out_nodes()) in [1, 2]

        hidden_shape = node.in_port(1).data.get_shape().copy()

        mark_input_bins(node, start_port=2)
        node.out_port(0).data.set_shape(hidden_shape)

        hidden_size = hidden_shape[1]
        if node.has_valid('hidden_size'):
            if node.hidden_size != hidden_size:
                raise Error("Input shape {} for hidden size doesn't match pre-defined hidden_size in node {}".format(
                    node.in_node(1).shape, node.soft_get('name')))
        else:
            node['hidden_size'] = hidden_size
