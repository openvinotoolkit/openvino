"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.common.partial_infer.utils import mark_input_bins
from mo.graph.graph import Node, Graph, add_opoutput
from mo.ops.op import Op


class RNN(Op):
    op = 'RNN'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': 'RNNSequence',  # should be never emitted to IR; for debugging purposes
            'op': __class__.op,
            'blobs_wrb': False,
            'has_num_directions': False,
            'direction': 'forward',
            'infer': __class__.infer,
            'multiplier': 1,
            'gate_order': np.array([0]),  # Only one gate in this cell
            'normalized': False,

            'activation_alpha': None,
            'activation_beta': None,
            'activations': None,
            'clip': None,
            'in_ports_count': 6,
            'out_ports_count': 2,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def supported_attrs():
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            # Additional attributes
            'activation_alpha',
            'activation_beta',
            'activations',
            'clip',
        ]

    def backend_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',  # one of 'forward', 'reverse', or 'bidirectional'
            'axis',

            # Additional attributes
            'activation_alpha',
            'activation_beta',
            ('activations', lambda node: ','.join(node.activations) if node.activations is not None else None),
            'clip',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) >= 3  # X, W and R
        assert len(node.in_nodes()) <= 5
        assert len(node.out_nodes()) <= 2

        rnn_infer(node, [1])


def rnn_infer(node: Node, out_ports=None):
    """
    General infer function for RNN, GRU, LSTM layers.
    Assume that 0-port input of node is input data for recurrent layer and node have attrs:
    hidden_size,
    """
    if out_ports is None:
        out_ports = []

    # 1. Necessary checks (from ONNX specification)
    assert node.batch_dim <= 1
    assert node.sequence_dim <= 1
    assert node.batch_dim != node.sequence_dim
    assert node.direction in ['forward', 'reverse', 'bidirectional']

    if node.blobs_wrb:
        mark_input_bins(node, ['W', 'R', 'B'])
    else:
        mark_input_bins(node)

    # 2. Output shape calculations
    input_shape = node.in_node(0).shape
    assert len(input_shape) == 3

    # Reshape input nodes
    for port in [2, 3]:
        if port in node.in_nodes() and len(node.in_node(port).in_nodes()) > 0 and \
                'zero_shapes' in node.in_node(port).in_node():
            for i in node.in_node(port).in_node().zero_shapes:
                if node.in_node(port).shape[i] != input_shape[i]:
                    node.in_node(port).value = np.repeat(node.in_node(port).value, input_shape[i], axis=i)
                    node.in_node(port).shape[i] = input_shape[i]

    out_shape = np.array([input_shape[node.sequence_dim], input_shape[node.batch_dim], node.hidden_size], dtype=np.int64)

    if node.batch_dim == 0:
        out_shape = np.array([input_shape[node.batch_dim], input_shape[node.sequence_dim], node.hidden_size], dtype=np.int64)

    num_directions = 2 if node.direction in ['bidirectional'] else 1
    if node.has_num_directions:
        if node.format == 'mxnet' and node.normalized is False:
            # In MXNet RNN layer return output with shape [seq_len, batch_size, hidden_size * num_directions]
            out_shape[-1] *= num_directions
        else:
            # ONNX-like, insert extra dimension to output shape for num_directions
            out_shape = np.insert(out_shape, 1, np.int64(num_directions))
    node.out_node(0).shape = out_shape

    # 3. Extra outputs for hidden/cell states shape calculations (optional)
    state_size = np.array([input_shape[node.batch_dim], node.hidden_size], dtype=np.int64)
    if node.has_num_directions:
        state_size = np.insert(state_size, 0, num_directions)

    if node.multilayers:
        # For multilayer case state sizes from every layer will be concatenated by last axis
        num_layers = node.num_layers
        state_size[-1] *= num_layers

    for i in out_ports:
        # If node hasn't consumers for hidden/cells state -> create them
        if i not in node.out_nodes():
            data_node = Op._create_data_node(
                node.graph,
                name=node.node + '/ExtraOutput/' + str(i),
                attrs={'executable': True}
            )
            if i not in node.out_ports():
                node.add_output_port(i)
            node.graph.add_edge(node.id, data_node.id, key=0, out=i)
            add_opoutput(node.graph, data_node.id, 0, False)
        else:
            data_node = node.out_node(i)
        data_node.shape = state_size.copy()
