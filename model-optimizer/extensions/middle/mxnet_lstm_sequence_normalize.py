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

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.reshape import Reshape
from mo.graph.graph import Node


class MXNetLSTMSequenceNormalize(MiddleReplacementPattern):
    ''' Convert blobs and shapes of MXNet-like LSTM to IE compatible form.

        The target form of this operation is not normally covered by a dedicated
        layer in IE. It should be further transformed to some other layer
        that are supported by IE. This transformation pass involves weights and
        shapes processing only.

        Post-conditions:

        Inputs have the following order:
            0: input data
            1: weights blob
            2: biases blob
            3: initial hidden state [optional]
            4: initial cell state [optional]
    '''
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(kind='op', op='LSTMSequence', format='mxnet')),
                ('input', dict(kind='data')),
                ('hidden_state', dict(kind='data')),
                ('cell_state', dict(kind='data')),
                ('params', dict(kind='data')),
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('hidden_state', 'lstm', {'in': 2}),
                ('cell_state', 'lstm', {'in': 3}),
                ('params', 'lstm', {'in': 1}),
            ]
        )


    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        input = match['input']
        lstm = match['lstm']
        params = match['params'].value.copy()
        hidden_state = match['hidden_state']
        cell_state = match['cell_state']

        hidden_state_edge_attrs = deepcopy(graph.get_edge_data(hidden_state.id, lstm.id)[0])
        cell_state_edge_attrs = deepcopy(graph.get_edge_data(cell_state.id, lstm.id)[0])

        graph.remove_edge(match['params'].id, lstm.id)
        graph.remove_edge(match['hidden_state'].id, lstm.id)
        graph.remove_edge(match['cell_state'].id, lstm.id)

        self.repack_weights(graph, input, lstm, params)

        reshape = Reshape(graph, dict(dim=[lstm.in_node(0).shape[0], lstm.hidden_size]))

        if len(lstm.in_nodes()) > 2:
            hidden_state_edge_attrs['in'] = 3
            new_init_h = reshape.create_node_with_data([hidden_state], attrs=dict(name=lstm.name + '/HiddenStateResize'))
            graph.add_edge(new_init_h.id, lstm.id, **hidden_state_edge_attrs)

        if len(lstm.in_nodes()) > 3:
            cell_state_edge_attrs['in'] = 4
            new_init_c = reshape.create_node_with_data([cell_state], attrs=dict(name=lstm.name + '/CellStateResize'))
            graph.add_edge(new_init_c.id, lstm.id, **cell_state_edge_attrs)


    def repack_weights(self, graph: nx.MultiDiGraph, input: Node, lstm: Node, params: np.array):
        input_size = input.shape[2]

        direction = 2 if lstm.has_num_directions else 1
        bsize = (2*lstm.hidden_size*direction*1)*4

        assert direction == 1

        W = np.array(params[0:len(params) - bsize])
        B = np.array(params[len(params) - bsize:])

        WX = np.array(W[0:lstm.hidden_size*4*input_size])
        WH = np.array(W[lstm.hidden_size*4*input_size:])

        WX = WX.reshape([lstm.hidden_size*4, input_size])
        WH = WH.reshape([lstm.hidden_size*4, lstm.hidden_size])

        WX = WX.transpose([1, 0])
        WH = WH.transpose([1, 0])

        WX = WX.reshape([
                1,  # 0: num of directions, limitation: should be 1
               -1,  # 3: input size
                4,  # 1: four output parts of the matrix for all gates in order: i, f, c, o
                lstm.hidden_size,  # 2: output size per direction and gate
        ])

        WH = WH.reshape([
                1,  # 0: num of directions, limitation: should be 1
               -1,  # 3: hidden state size
                4,  # 1: four output parts of the matrix for all gates in order: i, f, c, o
                lstm.hidden_size,  # 2: output size per direction and gate
        ])

        B = B.reshape([
                 1,  # 0: num of directions, limitation: should be 1
                 2,  # 3: num of component B
                 4,  # 1: four output parts of the matrix for all gates in order: i, f, c, o
                 lstm.hidden_size,  # 2: output size per direction and gate
        ])

        assert WX.shape[1] == input_size
        assert WH.shape[1] == lstm.hidden_size

        W = np.concatenate([WX, WH], axis=1)

        # Reorder gates: ifco --> fico
        gate_reorder = [1, 0, 2, 3]
        W = np.take(W, gate_reorder, axis=2)
        B = np.take(B, gate_reorder, axis=2)

        inout_reorder = [0, 2, 3, 1]
        W = W.transpose(inout_reorder)
        B = B.transpose(inout_reorder)

        final_shape = [W.shape[0] * W.shape[1] * lstm.hidden_size, -1]
        W = W.reshape(final_shape)
        B = B.reshape(final_shape)

        # Sum component of B
        B = np.add.reduce(B, axis=1, keepdims=True)
        B = B.squeeze(axis=1)

        assert W.ndim == 2
        assert B.ndim == 1
        assert W.shape[0] == lstm.hidden_size * 4
        assert B.shape[0] == lstm.hidden_size * 4
        assert W.shape[1] == lstm.hidden_size + input_size

        for blob, port, name in [(W, 1, 'weights'), (B, 2, 'biases')]:
            Op.create_and_connect_input_data_node(
                graph,
                lstm,
                {'value': blob, 'shape': np.array(blob.shape, dtype=np.int64)},
                {'in': port, 'bin': name, 'permutation': None}
            )
