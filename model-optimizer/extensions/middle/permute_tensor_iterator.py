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
import numpy as np
from copy import deepcopy

from mo.graph.graph import copy_node
from mo.utils.error import Error
from mo.middle.passes.eliminate import remove_op_node
from mo.middle.pattern_match import find_isomorphisms, find_pattern_matches
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from extensions.ops.lstm_sequence import LSTMSequence
from extensions.middle.FusePermutesSequence import FusePermutesSequence
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize, permute_before_and_after
from extensions.middle.lstm_sequence_tensor_iterator import LSTMSequenceTensorIterator


class PermuteTensorIteratorLSTM(MiddleReplacementPattern):
    ''' Fuses Permute(1,0,2) --> TI --> Permute(1,0,2) pattern to a single TI with changed axis.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    '''

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge, LSTMSequenceNormalize, LSTMSequenceTensorIterator, FusePermutesSequence]

    def pattern(self):
        return dict(
            nodes=[
                ('input'),
                ('direct_permute', dict(type='Permute')),
                ('input_permuted'),
                ('init_hidden'),
                ('init_cell'),

                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_permuted'),
                ('inverse_permute', dict(type='Permute')),
                ('output'),
            ],
            edges=[
                ('input', 'direct_permute'),
                ('direct_permute', 'input_permuted'),

                ('input_permuted', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('init_cell', 'ti', {'in': 2}),
                ('ti', 'output_permuted', {'out': 0}),

                ('output_permuted', 'inverse_permute'),
                ('inverse_permute', 'output'),
            ]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        nodes=[
            ('input_unsqueezed'),
            ('squeeze', dict(op='Reshape')),
            ('input_squeezed'),
            ('input_hidden'),
            ('input_cell'),
            ('weights'),
            ('biases'),

            ('lstm', dict(op='LSTMCell')),

            ('output_hidden'),
            ('output_cell'),
            ('unsqueeze', dict(op='Reshape')),
            ('output_unsqueezed'),
        ]
        edges=[
            ('input_unsqueezed', 'squeeze'),
            ('squeeze', 'input_squeezed'),

            ('input_squeezed', 'lstm', {'in': 0}),
            ('input_hidden', 'lstm', {'in': 1}),
            ('input_cell', 'lstm', {'in': 2}),
            ('weights', 'lstm', {'in': 3}),
            ('biases', 'lstm', {'in': 4}),

            ('lstm', 'output_hidden', {'out': 0}),
            ('lstm', 'output_cell', {'out': 1}),

            ('output_hidden', 'unsqueeze'),
            ('unsqueeze', 'output_unsqueezed'),
        ]
        ti = match['ti']
        isomorphisms = find_isomorphisms(ti.body, nodes, edges)
        if len(list(isomorphisms)) != 1:
            return

        direct_permute = match['direct_permute']
        inverse_permute = match['inverse_permute']

        if not direct_permute.has_valid('order') or not np.array_equal(direct_permute.order, [1, 0, 2]):
            return
        if not inverse_permute.has_valid('order') or not np.array_equal(inverse_permute.order, [1, 0, 2]):
            return

        # Remove permutes
        remove_op_node(graph, direct_permute)
        remove_op_node(graph, inverse_permute)
        match['output'].shape = match['output'].shape[[1, 0, 2]]

        # Modify axis in TI
        for port_map in [ti.input_port_map, ti.output_port_map]:
            for port in port_map:
                if 'axis' in port and port['axis'] is not None:
                    assert port['axis'] in [0, 1]
                    port['axis'] = 1 - port['axis']
