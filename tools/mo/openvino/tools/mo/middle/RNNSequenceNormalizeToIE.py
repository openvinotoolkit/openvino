# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_delete, mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices


class RNNSequenceNormalize(MiddleReplacementPattern):
    """
    This class normalize RNNSequence layers to IE-compatible from of weights, inputs and outputs.

    In this pass next will be done:
        1. Weights repack (squeeze all useless shapes in all blobs and concatenate W and R together, also add
                            bin param and all similar staff )
        1. UNSqueeze num directions (in states and )
        2. Initial states squeeze
        4. Renumbering inputs
        5. Ports checks

    After this normalization this layer will have next format of inputs:
            0: X input data,    shape [batch_size, seq_len, input_size]
            1: WR weights blob,  shape [M * hidden_size, hidden_size + input_size]
            2: B biases blob,   shape [M * hidden_size]
            3: (optional) sequence_length, shape [batch_size]
            4: initial hidden state, shape  [batch_size, hidden_size]
            5: (only for LSTM) initial cell state, shape [batch_size, hidden_size]
            6: (optional for LSTM) Peepholes weights, shape  [(M - 1) * hidden_size]

    """
    force_shape_inference = True

    def run_after(self):
        from openvino.tools.mo.middle.DecomposeBidirectionalRNNSequence import DecomposeBidirectionalRNNSequence
        return [DecomposeBidirectionalRNNSequence]

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', dict(kind='op', type='RNNSequence')),
                ('input', dict(kind='data')),
                ('W', dict(kind='data')),
                ('R', dict(kind='data')),
                ('B', dict(kind='data')),
            ],
            edges=[
                ('input', 'rnn_layer', {'in': 0}),
                ('W', 'rnn_layer', {'in': 1}),
                ('R', 'rnn_layer', {'in': 2}),
                ('B', 'rnn_layer', {'in': 3}),
            ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        self.repack_weights(graph, match)
        if match['rnn_layer'].has_num_directions:
            self.unsqueeze_num_directions(graph, match)
        self.squeeze_initial_states(graph, match)
        self.reordering_inputs(graph, match)
        # some additional checks for ports number and similar stuff

    def repack_weights(self, graph: Graph, match: dict):
        # Concat W, R in IE- format
        # Delete useless num_dir dimensions and n_cells dimensions in W, R, B (peepholes?)
        lstm = match['rnn_layer']
        W, R, B = match['W'].value.copy(), match['R'].value.copy(), match['B'].value.copy()

        graph.remove_edge(match['W'].id, lstm.id)
        graph.remove_edge(match['R'].id, lstm.id)
        graph.remove_edge(match['B'].id, lstm.id)

        # Sum component of B that correspond to W and R
        if lstm.op == 'GRU' and lstm.linear_before_reset:
            B_shape = mo_array(B.shape)
            B_shape[3] = 4
            B_shape[2] = 1
            B_tmp = np.zeros(shape=B_shape, dtype=np.float32)
            B_tmp[:, :, :, 0, :] = B[:, :, 0, 0, :] + B[:, :, 1, 0, :]
            B_tmp[:, :, :, 1, :] = B[:, :, 0, 1, :] + B[:, :, 1, 1, :]
            B_tmp[:, :, :, 2, :] = B[:, :, 0, 2, :][:, :, np.newaxis, :]
            B_tmp[:, :, :, 3, :] = B[:, :, 1, 2, :][:, :, np.newaxis, :]
            B = B_tmp
        else:
            B = np.sum(B, axis=2, keepdims=True)

        # Concatenate W, R to IE-compatible format
        assert len(W.shape) == 5
        assert len(R.shape) == 5
        WR = np.concatenate([W, R], axis=4)

        # Squeeze useless dimensions
        assert WR.shape[0] == 1  # num_dir == 1
        assert WR.shape[1] == 1  # num_cells == 1
        assert B.shape[0] == 1
        assert B.shape[1] == 1
        WR = WR.squeeze(axis=(0, 1))
        B = B.squeeze(axis=(0, 1))

        # Flatten all output (0, 1) and input dimensions (2, 3)
        final_shape_WR = [WR.shape[0] * WR.shape[1], -1]
        assert final_shape_WR[0] == lstm.hidden_size * lstm.multiplier
        WR = WR.reshape(final_shape_WR)

        final_shape_B = final_shape_WR
        if lstm.op == 'GRU' and lstm.linear_before_reset:
            final_shape_B[0] = lstm.hidden_size * 4
        B = B.reshape(final_shape_B)

        # Squeeze fake dimension in B
        B = B.squeeze(axis=-1)

        for blob, port, name in [(WR, 1, 'weights'), (B, 2, 'biases')]:
            Op.create_and_connect_input_data_node(
                graph,
                lstm,
                {'value': blob, 'shape': int64_array(blob.shape)},
                {'in': port, 'bin': name, 'permutation': None}
            )

    @staticmethod
    def unsqueeze_num_directions(graph: Graph, match: dict):
        """ Assuming considered LSTM/GRU/RNN node should has num_directions in output shape and add Unsqueeze
            to match it.
        """

        rnn_layer = match['rnn_layer']
        rnn_layer_name = rnn_layer.soft_get('name', rnn_layer.id)
        # num_directions is at 1st position in output shape, and in 0st position in hidden and cell states
        # please refer to docs in this transform

        direction_dim = [1, 0, 0]  # index of dimension with direction index
        for i in rnn_layer.out_nodes():
            old_data_node = rnn_layer.out_node(i)
            old_shape = old_data_node.shape.copy()
            new_shape = shape_delete(old_shape, direction_dim[i])

            data = Op._create_data_node(graph, name=rnn_layer.name + '/Out/{}/'.format(i), attrs={'shape': new_shape})
            graph.remove_edge(rnn_layer.id, old_data_node.id)
            graph.add_edge(rnn_layer.id, data.id, key=0, out=i)

            unsqueeze = Unsqueeze(graph, dict())

            unsqueeze_dim_data = Const(graph, {'name': rnn_layer.name + '/UnsqueezeNumDirections/{}/Dim'.format(i),
                                               'value': int64_array([direction_dim[i]])}).create_node_with_data()

            unsqueeze.create_node_with_data([data, unsqueeze_dim_data],
                                            dict(name=rnn_layer_name + '/UnsqueezeNumDirections/{}'.format(i)),
                                            data_nodes=[old_data_node])
    @staticmethod
    def squeeze_initial_states(graph: Graph, match: dict):
        """
        Squeeze input initial states of recurrent node to 2-D shape.
        """
        hidden_init_port = 5
        cell_init_port = 6

        rnn_layer = match['rnn_layer']
        # Add input ports to rnn_layer
        rnn_layer.add_sequence_of_ports(type='in', rng=range(7))
        rnn_layer_name = rnn_layer.soft_get('name', rnn_layer.id)

        assert hidden_init_port in rnn_layer.in_nodes()
        hidden_size = rnn_layer.hidden_size
        shape = Shape(graph, dict(name=rnn_layer_name + '/ShapeOf')).create_node()
        rnn_layer.in_port(0).get_source().connect(shape.in_port(0))

        reshape_h = create_op_node_with_second_input(graph, Reshape, second_input_value=int64_array([-1, hidden_size]),
                                                     op_attrs={'name': rnn_layer_name + '/HiddenStateResize',
                                                               'override_output_shape': True})
        rnn_layer.in_port(hidden_init_port).get_connection().insert_node(reshape_h)

        if rnn_layer.op == 'LSTM':
            assert cell_init_port in rnn_layer.in_nodes()
            reshape_c = create_op_node_with_second_input(graph, Reshape,
                                                         second_input_value=int64_array([-1, hidden_size]),
                                                         op_attrs={'name': rnn_layer_name + '/CellStateResize',
                                                                   'override_output_shape': True})
            rnn_layer.in_port(cell_init_port).get_connection().insert_node(reshape_c)

    @staticmethod
    def reordering_inputs(graph: Graph, match: dict):
        """
        Reorder (renumbering) inputs to described format. We need to renumber initial states ports.
        """
        rnn_layer = match['rnn_layer']
        assert 5 in rnn_layer.in_nodes()
        hidden_state_edge = graph.get_edge_data(rnn_layer.in_node(5).id, rnn_layer.id)
        hidden_state_edge[0]['in'] = 4

        if rnn_layer.op == 'LSTM':
            assert 6 in rnn_layer.in_nodes()
            cell_state_edge = graph.get_edge_data(rnn_layer.in_node(6).id, rnn_layer.id)
            cell_state_edge[0]['in'] = 5

    @staticmethod
    def ports_checks(graph: Graph, match: dict):
        """
            Check that all mandatory ports is present.
        """
        rnn_layer = match['rnn_layer']
        mandatory_ports = [0, 1, 2, 4]

        if rnn_layer.op == 'LSTM':
            mandatory_ports.append(5)

        assert set(rnn_layer.in_nodes().keys()) >= set(mandatory_ports)
