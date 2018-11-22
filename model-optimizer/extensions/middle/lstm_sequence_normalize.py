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

from mo.utils.error import Error
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape
from mo.graph.graph import Node


def inverse_perm(order: np.array):
    indices = np.empty(order.size, dtype=np.int64)
    indices[order] = np.arange(order.size)
    return indices


def permute_before_and_after(inp: Node, middle: Node, out: Node, order):
    ''' Insert two permutes: before middle node and after middle node.

        The first permute has a given order, the second permute has an
        inversed order.
    '''

    permute = Permute(middle.graph, dict(order=np.array(order)))

    edge_attrs = deepcopy(middle.graph.get_edge_data(inp.id, middle.id)[0])
    middle.graph.remove_edge(inp.id, middle.id)
    new_inp = permute.create_node_with_data([inp], dict(name=middle.name + '/InputPermute'))
    middle.graph.add_edge(new_inp.id, middle.id, **edge_attrs)

    permute = Permute(middle.graph, dict(order=inverse_perm(np.array(order))))

    middle.graph.remove_edge(middle.id, out.id)
    new_out = Op.create_data_node(middle.graph, middle, {'shape': out.shape[order]})
    permute.create_node_with_data([new_out], dict(name=middle.name + '/OutputPermute'), data_nodes=out)


class LSTMSequenceNormalize(MiddleReplacementPattern):
    ''' Convert blobs and shapes of ONNX-like LSTM to IE compatible form.

        Fuse W, R and optional B input blobs to weights and biases according
        to IE LSTM specification.

        The target form of this operation is not normally covered by a dedicated
        layer in IE. It should be further transformed to some other layer
        that are supported by IE. This transformation pass involves weights and
        shapes processing only.

        Post-conditions:

        Inputs have the forllowing order:
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
                ('lstm', dict(kind='op', op='LSTMSequence')),
                ('input', dict(kind='data')),
                ('W', dict(kind='data')),
                ('R', dict(kind='data')),
                # don't capture B here as it is optional, as well as extra outputs
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('W', 'lstm', {'bin': 'W'}),
                ('R', 'lstm', {'bin': 'R'}),
                ('lstm', 'output', {'out': 0}),
            ]
        )


    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        self.repack_weights(graph, match)
        self.batch_sequence_transpose(graph, match)
        self.check_not_supported_ports(graph, match)
        self.states_squeeze(graph, match)


    def repack_weights(self, graph: nx.MultiDiGraph, match: dict):

        lstm = match['lstm']
        W = match['W'].value.copy()
        R = match['R'].value.copy()

        graph.remove_edge(match['W'].id, lstm.id)
        graph.remove_edge(match['R'].id, lstm.id)

        # find optional 'B'
        if len(lstm.in_nodes()) > 3:
            # TODO: check if 'bin': 'B' attribute is assigned to this edge
            B = lstm.in_node(3).value.copy()
            graph.remove_edge(lstm.in_node(3).id, lstm.id)
        else:
            B = np.full([1, lstm.hidden_size*8], 0, dtype=np.float32)

        # Add extra dimensions for W, R and B for easier repacking

        B = B.reshape([
            1,  # 0: num of directions, limitation: should be 1
            2,  # 1: two input parts of the matrix: W, R
            4,  # 2: four output parts of the matrix for all gates in order: i, o, f, c
            lstm.hidden_size,  # 3: output size per direction and gate
            1,  # 4: fake dimension to match the input dimension in W and R for shorter code
        ])

        W, R = [x.reshape([
                1,  # 0: num of directions, limitation: should be 1
                1,  # 1: placeholder for concatenation of W and R matrices
                4,  # 2: four output parts of the matrix for all gates in order: i, o, f, c
                lstm.hidden_size,  # 3: output size per direction and gate
                -1])  # 4: input size
            for x in (W, R)]

        input_size = match['input'].shape[2]
        assert input_size == W.shape[-1]

        WR = np.concatenate([W, R], axis=1)

        # Reorder gates: iofc --> fico
        gate_reorder = [2, 0, 3, 1]
        WR = np.take(WR, gate_reorder, axis=2)
        B = np.take(B, gate_reorder, axis=2)

        # Sum component of B that correspond to W and R
        B = np.add.reduce(B, axis=1, keepdims=True)

        # Reorder dimensions by collection output dimensions first, then input dimension
        # Interpret the numbers below by looking at W, R and B reshape above in the code
        inout_reorder = [0, 2, 3, 1, 4]
        WR = WR.transpose(inout_reorder)
        B = B.transpose(inout_reorder)

        # Supposing it is unidirectional LSTM, squeeze 'direction' dimension
        assert WR.shape[0] == 1
        assert B.shape[0] == 1
        WR = WR.squeeze(axis=0)
        B = B.squeeze(axis=0)

        # Flatten all output (0, 1) and input dimensions (2, 3)
        final_shape = [WR.shape[0] * WR.shape[1], -1]
        WR = WR.reshape(final_shape)
        B = B.reshape(final_shape)

        # Squeeze fake dimension in B
        B = B.squeeze(axis=-1)

        assert WR.ndim == 2
        assert B.ndim == 1
        assert WR.shape[0] == lstm.hidden_size*4
        assert B.shape[0] == lstm.hidden_size*4
        assert WR.shape[1] == lstm.hidden_size + input_size

        for blob, port, name in [(WR, 1, 'weights'), (B, 2, 'biases')]:
            Op.create_and_connect_input_data_node(
                graph,
                lstm,
                {'value': blob, 'shape': np.array(blob.shape, dtype=np.int64)},
                {'in': port, 'bin': name, 'permutation': None}
            )


    def batch_sequence_transpose(self, graph: nx.MultiDiGraph, match: dict):

        lstm = match['lstm']
        inp = match['input']
        out = match['output']

        if lstm.batch_dim == 0:
            assert lstm.sequence_dim == 1
            # nothing to do -- it's already in normal form
            return

        assert lstm.sequence_dim == 0
        assert lstm.batch_dim == 1
        assert len(inp.shape) == 3

        # Reorder the first two dimensions on both ends: input and output.
        # Two Permute ops are inserted before and after the LSTM node.
        # In this transformation we don't analyze the rest of the model around
        # LSTM cell, so these Permute ops are not fused to some other layers here.
        # But other transformations in the pipeline may optimize the Permute ops out.

        lstm.batch_dim, lstm.sequence_dim = lstm.sequence_dim, lstm.batch_dim
        permute_before_and_after(inp, lstm, out, [1, 0, 2])


    def check_not_supported_ports(self, graph: nx.MultiDiGraph, match: dict):
        lstm = match['lstm']
        inputs = lstm.in_edges()
        assert 0 in inputs
        assert 1 in inputs and inputs[1]['bin'] == 'weights'
        assert 2 in inputs and inputs[2]['bin'] == 'biases'
        assert 3 not in inputs
        
        if not(set(list(inputs.keys())) <= set([0, 1, 2, 5, 6])):
            raise Error('Node {} that is interpreted as {} operation has '
                'some unexpected inputs initialized, '
                'they can include: sequence_lenght, '
                'and weight tensor for peepholes. '
                'This is not supported.'.format(lstm.name, lstm.op))


    def states_squeeze(self, graph: nx.MultiDiGraph, match: dict):

        lstm = match['lstm']

        reshape = Reshape(graph, dict(dim=[lstm.in_node(0).shape[0], lstm.hidden_size]))

        if len(lstm.in_nodes()) > 3:
            init_h = lstm.in_node(5)
            edge_attrs = deepcopy(graph.get_edge_data(init_h.id, lstm.id)[0])
            edge_attrs['in'] = 3
            graph.remove_edge(init_h.id, lstm.id)
            new_init_h = reshape.create_node_with_data([init_h], dict(name=lstm.name + '/HiddenStateResize'))
            graph.add_edge(new_init_h.id, lstm.id, **edge_attrs)

        if len(lstm.in_nodes()) > 4:
            init_c = lstm.in_node(6)
            edge_attrs = deepcopy(graph.get_edge_data(init_c.id, lstm.id)[0])
            edge_attrs['in'] = 4
            graph.remove_edge(init_c.id, lstm.id)
            new_init_c = reshape.create_node_with_data([init_c], dict(name=lstm.name + '/CellStateResize'))
            graph.add_edge(new_init_c.id, lstm.id, **edge_attrs)
