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

from mo.graph.graph import copy_node, Node, dict_includes
from mo.utils.error import Error
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.pattern_match import find_isomorphisms, find_pattern_matches
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from extensions.ops.lstm_sequence import LSTMSequence
from extensions.middle.FusePermutesSequence import FusePermutesSequence
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize, permute_before_and_after
from extensions.middle.lstm_sequence_tensor_iterator import LSTMSequenceTensorIterator
from extensions.middle.decompose_bi_lstm import DecomposeBiLSTM


class PermuteTensorIteratorLSTM(MiddleReplacementPattern):
    ''' Fuses Permute(1,0,2) --> TI --> Permute(1,0,2) pattern to a single TI with changed axis.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    '''

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge, LSTMSequenceNormalize, LSTMSequenceTensorIterator, FusePermutesSequence, DecomposeBiLSTM]

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

                ('input_permuted', 'ti', {'in': 0}),   # affected by permute
                ('init_hidden', 'ti', {'in': 1}),
                ('init_cell', 'ti', {'in': 2}),
                ('ti', 'output_permuted', {'out': 0}), # affected by permute

                ('output_permuted', 'inverse_permute'),
                ('inverse_permute', 'output'),
            ]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):

        # This transformation works if and only if a body of TI
        # matches the following topology (Reshape -> LSTMCell -> Reshape)
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
        isomorphism = isomorphisms[0]

        direct_permute = match['direct_permute']
        inverse_permute = match['inverse_permute']

        permute_order = [1, 0, 2]

        # Check both perumute orders exactly match expected one - [1, 0, 2]
        if not direct_permute.has_valid('order') or not np.array_equal(direct_permute.order, permute_order):
            return
        if not inverse_permute.has_valid('order') or not np.array_equal(inverse_permute.order, permute_order):
            return


        def find_ports(port_map: list, attrs: dict):
            """ Find all ports in a given port map with specified attributes """
            result = []
            for i, port in enumerate(port_map):
                if dict_includes(port, attrs):
                    result.append(i)
            return result

        # Check TI has only single partitioned input/output port; all partitioned ports have defined axis
        data_input_port = find_ports(ti.input_port_map, {'axis': lambda attr: attr in [0, 1]})
        data_output_port = find_ports(ti.output_port_map, {'axis': lambda attr: attr in [0, 1]})
        assert len(data_input_port) == 1
        assert len(data_output_port) == 1
        data_input_port = data_input_port[0]
        data_output_port = data_output_port[0]
        # Verify that they are really connected to Permute layers (guarantied by port numbers of TI, see the pattern)
        assert ti.in_edge(0)['external_port_id'] == ti.input_port_map[data_input_port]['external_port_id']
        assert ti.out_edge(0)['external_port_id'] == ti.output_port_map[data_output_port]['external_port_id']

        # Verify that the TI body have required Reshapes connected to the found ports
        squeeze = isomorphism['squeeze']
        unsqueeze = isomorphism['unsqueeze']
        assert squeeze['internal_layer_id'] == ti.input_port_map[data_input_port]['internal_layer_id']
        assert squeeze.in_edge(0)['internal_port_id'] == ti.input_port_map[data_input_port]['internal_port_id']
        assert unsqueeze['internal_layer_id'] == ti.output_port_map[data_output_port]['internal_layer_id']
        assert unsqueeze.out_edge(0)['internal_port_id'] == ti.output_port_map[data_output_port]['internal_port_id']
        assert len(squeeze.in_node().shape) == 3
        assert len(squeeze.out_node().shape) == 2
        assert len(unsqueeze.in_node().shape) == 2
        assert len(unsqueeze.out_node().shape) == 3

        # Remove permutes
        remove_op_node_with_data_node(graph, direct_permute)
        remove_op_node_with_data_node(graph, inverse_permute)
        match['output'].shape = match['output'].shape[permute_order]

        # swap 0/1 axis for partitioned ports
        ti.input_port_map[data_input_port]['axis'] = 1 - ti.input_port_map[data_input_port]['axis']
        ti.output_port_map[data_output_port]['axis'] = 1 - ti.output_port_map[data_output_port]['axis']

        # smap 0-th and 1-th shape entries for reshapes inside body
        squeeze.in_node().shape = squeeze.in_node().shape[[1, 0, 2]]
        unsqueeze.out_node().shape = unsqueeze.out_node().shape[[1, 0, 2]]
        unsqueeze.dim = unsqueeze.dim[[1, 0, 2]]
