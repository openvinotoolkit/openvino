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

from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape
from extensions.ops.lstm_cell import LSTMCell
from extensions.ops.tensor_iterator import TensorIterator
from extensions.middle.FusePermutesSequence import FusePermutesSequence


class LSTMSequenceTensorIterator(MiddleReplacementPattern):
    ''' Converts normalized LSTMSequence op to TensorIterator.

        Normalized LSTMSequence means that it should be processed by
        LSTMSequenceNormalize transform that ensures its stict form.

        This transformation builds an altenative sub-graph for LSTMSequence
        with TensorIterator connected in the same way as an original LSTMSequence
        node and with internal body represented as LSTMCell op node with necessary
        squeezes and unsqueezes around.
    '''

    enabled = True


    def run_after(self):
        return [LSTMSequenceNormalize]


    def run_before(self):
        return [FusePermutesSequence]


    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(kind='op', op='LSTMSequence')),
                ('input', dict(kind='data')),
                ('weights', dict(kind='data')),
                ('biases', dict(kind='data')),
                # don't capture optional input initial states here
                ('output', dict(kind='data')),
                # don't capture optional output last states here
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('weights', 'lstm', {'bin': 'weights', 'in': 1}),
                ('biases', 'lstm', {'bin': 'biases', 'in': 2}),
                ('lstm', 'output', {'out': 0}),
            ]
        )


    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        lstm = match['lstm']

        # Build TensorIterator body first
        body = nx.MultiDiGraph(name=lstm.name + '/sub_graph')
        inputs = [Op._create_data_node(body, lstm.name + '/inport/' + str(inp), {'shape': lstm.in_node(inp).shape.copy(), 'value': lstm.in_node(inp).value.copy() if lstm.in_node(inp).value is not None else None}) for inp in [0, 3, 4, 1, 2]]
        inputs[0].shape[lstm.sequence_dim] = 1
        input_squeeze = Reshape(body, dict(name=lstm.name + '/input_squeeze', dim=np.delete(inputs[0].shape, lstm.sequence_dim), internal_layer_id=0))
        inputs[0] = input_squeeze.create_node_with_data([inputs[0]], edge_attrs=[{'internal_port_id': 0}])
        lstm_cell_op = LSTMCell(body, dict(hidden_size=match['lstm'].hidden_size, name=lstm.name + '/LSTMCell', internal_layer_id=1))
        outputs = [Op._create_data_node(body, lstm.name + '/outport/' + str(out),  {'shape': lstm.out_node(out).shape.copy() if out in lstm.out_nodes() else lstm.in_node(3).shape.copy()}) for out in [0,1] ]
        unsqueezed_output_shape = outputs[0].shape.copy()
        unsqueezed_output_shape[lstm.sequence_dim] = 1
        squeezed_output_shape = np.delete(unsqueezed_output_shape, lstm.sequence_dim)
        outputs[0].shape = squeezed_output_shape
        output_unsqueeze = Reshape(body, dict(name=lstm.name + 'output_unsqueeze', dim=unsqueezed_output_shape, internal_layer_id=2))
        # TODO edge attributes should be assigned by the op itself
        lstm_cell_node = lstm_cell_op.create_node_with_data(inputs, data_nodes=outputs, edge_attrs=[{}, {'internal_port_id': 1}, {'internal_port_id': 2}, {'bin': 'weights'}, {'bin': 'biases'}])
        lstm_cell_node[0].in_node().out_edge(0)['internal_port_id'] = 4
        lstm_cell_node[0].in_node().out_edge(1)['internal_port_id'] = 5
        lstm_cell_node[0] = output_unsqueeze.create_node_with_data([lstm_cell_node[0]])
        lstm_cell_node[0].in_node().out_edge(0)['internal_port_id'] = 3
        
        ti_op = TensorIterator(graph, {
            'name': lstm.name + '/TensorIterator',
            'body': body,

            # FOR TESTING PURPOSES
            'input_port_map': [
                {
                    'external_port_id': 0,
                    'internal_layer_id': 0,
                    'internal_port_id': 0,
                    'axis': lstm.sequence_dim,
                    'stride': 1,
                    'part_size': 1,
                },
                {
                    'external_port_id': 1,
                    'internal_layer_id': 1,
                    'internal_port_id': 1,
                },
                {
                    'external_port_id': 2,
                    'internal_layer_id': 1,
                    'internal_port_id': 2,
                },
            ],

            'output_port_map': [
                {
                    'external_port_id': 3,
                    'internal_layer_id': 2,
                    'internal_port_id': 3,
                    'axis': lstm.sequence_dim,
                    'stride': 1,
                    'part_size': 1,
                },
            ],
            'back_edges': [
                {
                    'from_layer': 1,
                    'from_port': 4,
                    'to_layer': 1,
                    'to_port': 1,
                },
                {
                    'from_layer': 1,
                    'from_port': 5,
                    'to_layer': 1,
                    'to_port': 2,
                },
            ]
        })

        outs = ti_op.create_node_with_data([lstm.in_node(i) for i in [0, 3, 4]], data_nodes=list(lstm.out_nodes().values()), edge_attrs=[{'external_port_id': 0}, {'external_port_id': 1}, {'external_port_id': 2}])
        graph.remove_node(lstm.id)
        outs.in_edge(0)['external_port_id'] = 3
