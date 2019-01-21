"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from mo.middle.replacement import MiddleReplacementPattern
from extensions.ops.lstm_sequence import LSTMSequence
from extensions.middle.FusePermutesSequence import FusePermutesSequence
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize
from extensions.middle.lstm_sequence_tensor_iterator import LSTMSequenceTensorIterator
from extensions.middle.permute_tensor_iterator import PermuteTensorIteratorLSTM
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class ReverseTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Reverse operations around TI: ReverseSequence --> TI  --> ReverseSequence.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True

    def run_after(self):
        return [
            TensorIteratorMerge,
            LSTMSequenceNormalize,
            LSTMSequenceTensorIterator,
            FusePermutesSequence,
            PermuteTensorIteratorLSTM,
        ]

    def pattern(self):
        return dict(
            nodes=[
                ('input'),
                ('direct_reverse', dict(op='ReverseSequence')),
                ('input_reversed'),
                ('init_hidden'),
                ('init_cell'),

                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_reversed'),
                ('inverse_reverse', dict(op='ReverseSequence')),
                ('output'),
            ],
            edges=[
                ('input', 'direct_reverse', {'in': 0}),
                ('direct_reverse', 'input_reversed'),

                ('input_reversed', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('init_cell', 'ti', {'in': 2}),
                ('ti', 'output_reversed', {'out': 0}),

                ('output_reversed', 'inverse_reverse', {'in': 0}),
                ('inverse_reverse', 'output'),
            ]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        ti = match['ti']
        direct_reverse = match['direct_reverse']
        inverse_reverse = match['inverse_reverse']

        assert direct_reverse.seq_dim == inverse_reverse.seq_dim
        assert direct_reverse.batch_dim is None and inverse_reverse.batch_dim is None or \
            direct_reverse.batch_dim == inverse_reverse.batch_dim

        # Modify stride in TI
        for port_map in [ti.input_port_map, ti.output_port_map]:
            for port in port_map:
                if 'axis' in port and port['axis'] is not None and 'external_port_id' in port:
                    assert port['axis'] == direct_reverse.seq_dim, \
                        'axis == {} != {} == direct_reverse.seq_dim'.format(port['axis'], direct_reverse.seq_dim)
                    if 'stride' not in port or port['stride'] is None:
                        port['stride'] = 1
                    assert port['stride'] in [-1, 1]
                    port['stride'] = -port['stride']
                    if port['stride'] == -1:
                        port['start'] = -1
                        port['end'] = 0
                    elif port['stride'] == 1:
                        port['start'] = None
                        port['end'] = None

        # Remove reverses
        remove_op_node_with_data_node(graph, direct_reverse)
        remove_op_node_with_data_node(graph, inverse_reverse)
