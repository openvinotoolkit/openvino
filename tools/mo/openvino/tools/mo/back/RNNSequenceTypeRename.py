# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class RNNSequence(BackReplacementPattern):
    """
    This transform change type RNNSequence (internal MO type for all recurrent layers)
    to correct operation name.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', {'type': 'RNNSequence'})
            ],
            edges=[]
        )

    _supported_ops = ['RNN', 'LSTM', 'GRU']

    def replace_pattern(self, graph: Graph, match: dict):
        rnn_layer = match['rnn_layer']
        assert rnn_layer['op'] in self._supported_ops
        rnn_layer['type'] = rnn_layer['op'] + 'Sequence'
