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

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class MXNetRNNSequenceNormalize(MiddleReplacementPattern):
    """
        Convert blobs and shapes of MXNet-like RNN cell to IE compatible form.

        The target form of this operation is not normally covered by a dedicated
        layer in IE. It should be further transformed to some other layer
        that are supported by IE. This transformation pass involves weights and
        shapes processing only.

        Post-conditions:
        Inputs:
            0: X input data,    shape [batch_size, seq_len, input_size] (or [seq_len. bathc_size, int_size], depends on
                                batch_dim param)
            1: W weights blob,  shape [num_dir, n_cells, M, hidden_size, input_size]
            2: R weights blob,  shape [num_dir, n_cells, M, hidden_size, hidden_size]
            3: B biases blob,   shape [num_dir, n_cells, 2, M, hidden_size]
            4: (optional) sequence_length, shape [batch_size]
            5: initial hidden state, shape  [num_dir, batch_size, hidden_size]
                                                      ([num_dir, n_cells, batch_size, hidden_size] if num_cells != 1)
            6: (only for LSTM) initial cell state, shape [num_dir, batch_size, hidden_size]
            7: (optional for LSTM) Peepholes weights, shape  [num_dir, n_cells, (M - 1) * hidden_size]

        Outputs:
            0: Y output blob,  shape [batch_size, num_dir, seq_len, hidden_size]
            1: (optional) Y_h, shape [num_dir, batch_size, hidden_size]
            2: (optional for LSTM) Y_c, shape [num_dir, batch_size, hidden_size]

        Where:
            M -- number of gates in this cell (4 for LSTM, 3 for GRU, 1 for RNN).
            num_dir -- number of directions ('forvard', 'bidirectional', 'reverse')
            n_cells -- number of cells in layer (always 1 for ONNX).

    """
    enabled = True

    def run_after(self):
        from extensions.middle.MXNetSplitMultiLayers import MXNetSplitLayersToRNNSequence
        return [MXNetSplitLayersToRNNSequence]

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', dict(kind='op', type='RNNSequence', format='mxnet')),
                ('input', dict(kind='data')),
                ('params', dict(kind='data')),
            ],
            edges=[
                ('input', 'rnn_layer', {'in': 0}),
                ('params', 'rnn_layer', {'in': 1}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        rnn_layer = match['rnn_layer']

        self.check_init_states(graph, match)
        self.repack_weights(graph, match)
        self.add_output_reshape(graph, match)
        self.check_input_ports(graph, match)
        rnn_layer['normalized'] = True

    @staticmethod
    def repack_weights(graph: Graph, match: dict):
        input = match['input']
        rnn_layer = match['rnn_layer']
        params = match['params'].value.copy()

        graph.remove_edge(match['params'].id, rnn_layer.id)

        input_size = input.shape[2]
        direction = 2 if rnn_layer.has_num_directions else 1
        bsize = (2 * rnn_layer.hidden_size * direction * 1) * rnn_layer.multiplier

        W = np.array(params[0:len(params) - bsize])
        B = np.array(params[len(params) - bsize:])

        W = W.reshape((direction, -1))
        B = B.reshape((direction, -1))

        W, R = np.array(W[:, 0:rnn_layer.hidden_size * rnn_layer.multiplier * input_size]), np.array(W[:, rnn_layer.hidden_size * rnn_layer.multiplier* input_size:])

        W, R = [x.reshape([
            direction,  # 0: num of directions
            1,  # 1: num_cells
            rnn_layer.multiplier,  # 2: four output parts of the matrix for all gates
            rnn_layer.hidden_size,  # 3: output size per direction and gate
            -1])  # 4: input size/hidden size in W/R correspondingly
            for x in (W, R)]

        assert W.shape[-1] == input_size
        assert R.shape[-1] == rnn_layer.hidden_size

        B = B.reshape([
                 direction,  # 0: num of directions, limitation: should be 1
                 1,
                 2,  # 3: num of component B
                 rnn_layer.multiplier,  # 1: four output parts of the matrix for all gates in order: i, f, c, o
                 rnn_layer.hidden_size,  # 2: output size per direction and gate
        ])

        # Reorder gates: ifco --> fico
        gate_reorder = rnn_layer.gate_order
        W = np.take(W, gate_reorder, axis=2)
        R = np.take(R, gate_reorder, axis=2)
        B = np.take(B, gate_reorder, axis=3)

        # Add ports to rnn_layer
        rnn_layer.add_sequence_of_ports(type='in', rng=range(7))

        for blob, port in [(W, 1), (R, 2), (B, 3)]:
            Op.create_and_connect_input_data_node(
                graph,
                rnn_layer,
                {'value': blob, 'shape': np.array(blob.shape, dtype=np.int64)},
                {'in': port, 'permutation': None}
            )

    @staticmethod
    def check_init_states(graph: Graph, match: dict):
        """
        Check if cell have initial states and create zeros states if not.
        And renumber ports for this states.
        """
        rnn_cell = match['rnn_layer']
        num_directions = 2 if rnn_cell.direction == 'bidirectional' else 1
        batch_size = rnn_cell.in_node(0).shape[rnn_cell.batch_dim]

        h_init_port = 5
        c_init_port = 6

        if 2 not in rnn_cell.in_nodes():
            h_shape = [num_directions, batch_size, rnn_cell.hidden_size]  # from ONNX spec
            h_init = np.full(h_shape, 0, dtype=np.float32)
            Op.create_and_connect_input_data_node(
                graph,
                rnn_cell,
                {'value': h_init, 'shape': np.array(h_init.shape, dtype=np.int64)},
                {'in': h_init_port, 'permutation': None}
            )
        else:
            hidden_state_edge = graph.get_edge_data(rnn_cell.in_node(2).id, rnn_cell.id)
            hidden_state_edge[0]['in'] = h_init_port

        if rnn_cell.op == 'LSTM':
            if 3 not in rnn_cell.in_nodes():
                c_shape = [num_directions, batch_size, rnn_cell.hidden_size]  # from ONNX spec
                c_init = np.full(c_shape, 0, dtype=np.float32)
                Op.create_and_connect_input_data_node(
                    graph,
                    rnn_cell,
                    {'value': c_init, 'shape': np.array(c_init.shape, dtype=np.int64)},
                    {'in': c_init_port, 'permutation': None}
                )
            else:
                cell_state_edge = graph.get_edge_data(rnn_cell.in_node(3).id, rnn_cell.id)
                cell_state_edge[0]['in'] = c_init_port

    @staticmethod
    def add_output_reshape(graph: Graph, match: dict):
        """
        Since MXNet Y output shape is [batch_size, seq_len, hidden_size * num_directions] we need to add reshape
        from above common format [batch_size, num_directions, seq_len, hidden_size] to MXNet format.
        """
        lstm = match['rnn_layer']
        input = match['input']
        if not lstm.has_num_directions:
            return
        old_data_node =lstm.out_node(0)
        num_directions = 2 if lstm.direction in ['bidirectional'] else 1
        mxnet_shape = lstm.out_node(0).shape.copy()

        if lstm.batch_dim == 0:
            mo_shape = np.array([input.shape[lstm.batch_dim], input.shape[lstm.sequence_dim], lstm.hidden_size],
                             dtype=np.int64)
        else:
            mo_shape = np.array([input.shape[lstm.sequence_dim], input.shape[lstm.batch_dim], lstm.hidden_size],
                                dtype=np.int64)

        if lstm.has_num_directions:
            mo_shape = np.insert(mo_shape, 1, np.int64(num_directions))

        new_data = Op._create_data_node(graph, name=lstm.name + '/Data/Reshape_mxnet/', attrs={'shape': mo_shape})
        graph.remove_edge(lstm.id, old_data_node.id)
        graph.add_edge(lstm.id, new_data.id, key=0, out=0)

        # Add Transpose
        permute_order = Const(graph, dict(value=int64_array([0, 2, 1, 3]))).create_node_with_data()
        permute_data = Transpose(graph, dict(name=lstm.name + '/Transpose_mxnet/')
                                 ).create_node_with_data([new_data, permute_order])

        # Add Reshape
        reshape = Reshape(graph, dict(name=lstm.name + '/Reshape_mxnet/'))
        reshape_dim_data = Const(graph, {'name': lstm.name + '/Reshape_mxnet_dim',
                                         'value': mxnet_shape}).create_node_with_data()

        reshape.create_node_with_data([permute_data, reshape_dim_data], dict(), data_nodes=[old_data_node])

    @staticmethod
    def check_input_ports(graph: Graph, match: dict):
        """
        Check that all mandatory ports is present.
        """
        rnn_layer = match['rnn_layer']
        mandatory_ports = [0, 1, 2, 3, 5]

        if rnn_layer.op == 'LSTM':
            mandatory_ports.append(6)

        assert set(rnn_layer.in_nodes().keys()) >= set(mandatory_ports)
