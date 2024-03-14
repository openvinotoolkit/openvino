# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.RNN import rnn_infer, RNN
from openvino.tools.mo.ops.op import Op


class GRU(Op):
    op = 'GRU'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': 'RNNSequence',  # should be never emitted to IR; for debugging purposes
            'op': __class__.op,
            'blobs_wrb': False,
            'has_num_directions': False,
            'direction': 'forward',
            'infer': __class__.infer,
            'reverse_infer': RNN.reverse_infer,
            'multiplier': 3,
            'multilayers': False,
            'gate_order': mo_array([0, 1, 2]),  # TODO: change it later
            'normalized': False,

            'activation_alpha': None,
            'activation_beta': None,
            'activations': None,
            'clip': None,
            'linear_before_reset': None,
            'in_ports_count': 6,
            'out_ports_count': 2,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def supported_attrs():
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            'activation_alpha',
            'activation_beta',
            'activations',
            'clip',
            'linear_before_reset',
        ]

    def backend_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            ('activations', lambda node: ','.join(node['activations']) if node.has_and_set('activations') else None),
            ('activations_alpha', lambda node: ','.join(map(str, node['activations_alpha']))
                if node.has_and_set('activations_alpha') else None),
            ('activations_beta', lambda node: ','.join(map(str, node['activations_beta']))
                if node.has_and_set('activations_beta') else None),
            'clip',
            'linear_before_reset',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) >= 3  # X, W and R
        assert len(node.in_nodes()) <= 5
        assert len(node.out_nodes()) <= 2

        rnn_infer(node, [1])
