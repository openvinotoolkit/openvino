"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


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
