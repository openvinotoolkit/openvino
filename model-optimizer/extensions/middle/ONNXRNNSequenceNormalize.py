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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op


class ONNXRNNSequenceNormalize(MiddleReplacementPattern):
    """
        Convert blobs and shapes of ONNX-like LSTM, GRU, RNN cells to common form (internal for MO).
        After this normalization pass passes for spliting bidirectional calls and
        multilayer cells will be applied.

        This transformation pass involves weights and shapes processing only:
            1. Weights reshaping and reordering
            2. Gates reordering


        Inputs will have the following order after normalising:
            0: X input data,    shape [batch_size, seq_len, input_size]
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

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', dict(kind='op', type='RNNSequence', format='onnx')),
                ('input', dict(kind='data')),
                ('W', dict(kind='data')),
                ('R', dict(kind='data')),
            ],
            # We are not handling optional inputs
            edges=[
                ('input', 'rnn_layer', {'in': 0}),
                ('W', 'rnn_layer', {'bin': 'W'}),
                ('R', 'rnn_layer', {'bin': 'R'}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        self.repack_weights(graph, match)
        self.check_init_states(graph, match)
        self.check_input_ports(graph, match)
        match['rnn_layer']['normalized'] = True

    @staticmethod
    def repack_weights(graph: Graph, match: dict):
        """
        Repack weights into general format (described above) and reorder gates.
        """
        rnn_layer = match['rnn_layer']
        W = match['W'].value.copy()
        R = match['R'].value.copy()
        num_directions = 2 if rnn_layer.direction == 'bidirectional' else 1

        graph.remove_edge(match['W'].id, rnn_layer.id)
        graph.remove_edge(match['R'].id, rnn_layer.id)

        # find optional 'B' biases blob
        if 3 in rnn_layer.in_nodes():
            # TODO: check if 'bin': 'B' attribute is assigned to this edge
            B = rnn_layer.in_node(3).value.copy()
            graph.remove_edge(rnn_layer.in_node(3).id, rnn_layer.id)
        else:
            B_shape = [num_directions, 2 * rnn_layer.multiplier * rnn_layer.hidden_size]  # from ONNX spec
            B = np.full(B_shape, 0, dtype=np.float32)

        # Add extra dimensions for W, R and B for easier repacking and reordering
        B = B.reshape([
            num_directions,  # 0: num of directions
            rnn_layer.num_layers,  # 1: num_layers
            2,  # 2: two input parts of the matrix: W, R
            rnn_layer.multiplier,  # 3: four output parts of the matrix for all gates in order: i, o, f, c
            rnn_layer.hidden_size,  # 4: output size per direction and gate
        ])

        W, R = [x.reshape([
                num_directions,  # 0: num of directions
                rnn_layer.num_layers,  # 1: num_layers
                rnn_layer.multiplier,  # 2: four output parts of the matrix for all gates in order: i, o, f, c
                rnn_layer.hidden_size,  # 3: output size per direction and gate
                -1])  # 4: input size/hidden size in W/R correspondingly
                for x in (W, R)]

        input_size = match['input'].shape[2]
        assert input_size == W.shape[-1]

        # Reorder gates: iofc --> fico
        gate_reorder = rnn_layer.gate_order
        W, R = (np.take(x, gate_reorder, axis=2) for x in (W, R))
        B = np.take(B, gate_reorder, axis=3)

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
        """
        rnn_layer = match['rnn_layer']
        num_directions = 2 if rnn_layer.direction == 'bidirectional' else 1
        batch_size = rnn_layer.in_node(0).shape[rnn_layer.batch_dim]

        h_init_port = 5
        c_init_port = 6

        if h_init_port not in rnn_layer.in_nodes():
            h_shape = [num_directions, batch_size, rnn_layer.hidden_size]  # from ONNX spec
            h_init = np.full(h_shape, 0, dtype=np.float32)
            Op.create_and_connect_input_data_node(
                graph,
                rnn_layer,
                {'value': h_init, 'shape': np.array(h_init.shape, dtype=np.int64)},
                {'in': h_init_port, 'permutation': None}
            )

        if rnn_layer.op == 'LSTM':
            if c_init_port not in rnn_layer.in_nodes():
                c_shape = [num_directions, batch_size, rnn_layer.hidden_size]  # from ONNX spec
                c_init = np.full(c_shape, 0, dtype=np.float32)
                Op.create_and_connect_input_data_node(
                    graph,
                    rnn_layer,
                    {'value': c_init, 'shape': np.array(c_init.shape, dtype=np.int64)},
                    {'in': c_init_port, 'permutation': None}
                )

    @staticmethod
    def check_input_ports(graph: Graph, match: dict):
        """
        Check that all mandatory ports is present.
        """
        rnn_layer = match['rnn_layer']
        mandatory_ports = [0, 1, 2, 3, 5]

        if rnn_layer.op == 'LSTM':
            mandatory_ports.extend([6])

        assert set(rnn_layer.in_nodes().keys()) >= set(mandatory_ports)
