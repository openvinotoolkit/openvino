"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from extensions.middle.LSTMRNNSequenceToTensorIterator import LSTMToTensorIterator
from extensions.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
from extensions.middle.SwapAxesMiddleReplacer import SwapAxisMiddleReplacer
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from mo.graph.graph import dict_includes, Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.pattern_match import find_isomorphisms
from mo.middle.replacement import MiddleReplacementPattern


class TransposeTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Transpose(1,0,2) --> TI --> Transpose(1,0,2) pattern to a single TI with changed axis.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge, ONNXRNNSequenceNormalize, LSTMToTensorIterator, SwapAxisMiddleReplacer]

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('direct_permute', dict(kind='op', op='Transpose')),
                ('input_permuted', dict(kind='data')),
                ('init_hidden', dict(kind='data')),
                ('init_cell', dict(kind='data')),
                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_permuted', dict(kind='data')),
                ('inverse_permute', dict(op='Transpose')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'direct_permute'),
                ('direct_permute', 'input_permuted'),

                ('input_permuted', 'ti', {'in': 0}),  # affected by permute
                ('init_hidden', 'ti', {'in': 1}),
                ('init_cell', 'ti', {'in': 2}),
                ('ti', 'output_permuted', {'out': 0}),  # affected by permute

                ('output_permuted', 'inverse_permute'),
                ('inverse_permute', 'output'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):

        # This transformation works if and only if a body of TI
        # matches the following topology (Reshape -> LSTMCell -> Reshape)
        nodes = [
            ('squeeze_dim',  dict(kind='op', op='Const')),
            ('squeeze_dim_data',  dict(kind='data')),

            ('unsqueeze_dim', dict(kind='op', op='Const')),
            ('unsqueeze_dim_data', dict(kind='data')),

            ('input_unsqueezed', dict(kind='data')),
            ('squeeze', dict(kind='op', op='Reshape')),
            ('input_squeezed', dict(kind='data')),
            ('input_hidden', dict(kind='data')),
            ('input_cell', dict(kind='data')),
            ('weights', dict(kind='data')),
            ('biases', dict(kind='data')),

            ('lstm', dict(kind='op', op='LSTMCell')),

            ('output_hidden', dict(kind='data')),
            ('output_cell', dict(kind='data')),
            ('unsqueeze', dict(kind='op', op='Reshape')),
            ('output_unsqueezed', dict(kind='data')),

            ('const_w', dict(kind='op', op='Const')),
            ('const_b', dict(kind='op', op='Const')),

            ('op_output', dict(kind='op', op='Result')),
            ('op_output_1', dict(kind='op', op='Result')),
            ('op_output_2', dict(kind='op', op='Result'))

        ]
        edges = [
            ('input_unsqueezed', 'squeeze', {'in': 0}),
            ('squeeze', 'input_squeezed'),

            ('squeeze_dim', 'squeeze_dim_data'),
            ('squeeze_dim_data', 'squeeze', {'in': 1}),

            ('input_squeezed', 'lstm', {'in': 0}),
            ('input_hidden', 'lstm', {'in': 1}),
            ('input_cell', 'lstm', {'in': 2}),
            ('weights', 'lstm', {'in': 3}),
            ('biases', 'lstm', {'in': 4}),

            ('const_w', 'weights'),
            ('const_b', 'biases'),

            ('lstm', 'output_hidden', {'out': 0}),
            ('lstm', 'output_cell', {'out': 1}),

            ('output_hidden', 'unsqueeze'),
            ('unsqueeze', 'output_unsqueezed'),

            ('unsqueeze_dim', 'unsqueeze_dim_data'),
            ('unsqueeze_dim_data', 'unsqueeze', {'in': 1}),

            ('output_unsqueezed', 'op_output'),
            ('output_hidden', 'op_output_1'),
            ('output_cell', 'op_output_2'),

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
        direct_order = direct_permute.in_port(1).data.get_value()
        if direct_order is None or not np.array_equal(direct_order, permute_order):
            return
        inverse_order = inverse_permute.in_port(1).data.get_value()
        if inverse_order is None or not np.array_equal(inverse_order, permute_order):
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
        # Verify that they are really connected to Transpose layers (guarantied by port numbers of TI, see the pattern)
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

        unsqueeze_dim = isomorphism['unsqueeze_dim']
        unsqueeze_dim_data = isomorphism['unsqueeze_dim_data']

        unsqueeze_dim.value = unsqueeze_dim.value[[1, 0, 2]]
        unsqueeze_dim_data.value = unsqueeze_dim_data.value[[1, 0, 2]]
