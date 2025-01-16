# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins, shape_insert, dynamic_dimension, \
    shape_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph, add_opoutput, Error
from openvino.tools.mo.ops.op import Op


class RNN(Op):
    op = 'RNN'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': 'RNNSequence',  # should be never emitted to IR; for debugging purposes
            'op': self.op,
            'blobs_wrb': False,
            'has_num_directions': False,
            'direction': 'forward',
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,
            'multiplier': 1,
            'gate_order': mo_array([0]),  # Only one gate in this cell
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

    @staticmethod
    def reverse_infer(node: Node):
        if node.in_port(0).data.get_shape() is not None:
            return

        input_size = get_rnn_input_size(node)
        batch_size, seq_len = get_rnn_batch_size_and_seq_len(node)
        # ONNX has the same input layout
        input_shape = shape_array([seq_len, batch_size, input_size])
        if node.format == 'tf':
            input_shape = shape_array([batch_size, seq_len, input_size])

        node.in_port(0).data.set_shape(input_shape)


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

    out_shape = [input_shape[node.sequence_dim], input_shape[node.batch_dim], node.hidden_size]

    if node.batch_dim == 0:
        out_shape = [input_shape[node.batch_dim], input_shape[node.sequence_dim], node.hidden_size]

    num_directions = 2 if node.direction in ['bidirectional'] else 1
    if node.has_num_directions:
        # ONNX-like, insert extra dimension to output shape for num_directions
        out_shape = shape_insert(out_shape, 1, np.int64(num_directions))

    # 0 output is required creating it if doesn't exist
    if 0 not in node.out_nodes():
        data_node = Op._create_data_node(
            node.graph,
            name=node.node + '/ExtraOutput/{}'.format(0),
            attrs={'executable': True}
        )
        if 0 not in node.out_ports():
            node.add_output_port(0)
        node.graph.add_edge(node.id, data_node.id, key=0, out=0)
        add_opoutput(node.graph, data_node.id, 0, False)
    node.out_port(0).data.set_shape(out_shape)

    # 3. Extra outputs for hidden/cell states shape calculations (optional)
    state_size = [input_shape[node.batch_dim], node.hidden_size]
    if node.has_num_directions:
        state_size = shape_insert(state_size, 0, num_directions)

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
        data_node.shape = shape_array(state_size)


def get_rnn_batch_size_and_seq_len(node: Node):
    """
    Gets batch_size and sequence_length from RNN constant inputs
    and output shapes retrieved during reverse_infer

    :param node:
    :return:
    """
    node_name = node.soft_get('name', node.id)
    out_shape = node.out_port(0).data.get_shape()
    batch_size = dynamic_dimension
    seq_len = dynamic_dimension
    in_port_with_initial_states = 3  # initial hidden size values is framework dependent

    if out_shape is not None:
        # note that op is not in opset state but in the state of the original framework
        if node.batch_dim == 1:
            seq_len = out_shape[0]

            if node.format == 'onnx':
                assert len(out_shape) == 4, 'incorrect out_shape rank for node {}'.format(node_name)
                # even for ONNX in extractor 'batch_dim': 1 (front/onnx/lstm_ext.py:26) despite the fact that
                # out_shape = [seq_len, num_directions, batch_size, hidden_size]
                batch_size = out_shape[2]
                in_port_with_initial_states = 5
            elif node.format == 'tf':
                log.error('reverse infer for TensorFlow RNN operation {} is not implemented yet'.format(node_name),
                          extra={'is_warning': True})
            else:
                raise Error('Incorrect framework name')
        elif node.batch_dim == 0:
            # out_shape = [batch_size, num_directions, seq_len, hidden_size]
            batch_size = out_shape[0]
            seq_len = out_shape[2]
            in_port_with_initial_states = 3
        else:
            raise Error('incorrect batch_dim for node {}'.format(node_name))

    if batch_size is dynamic_dimension:
        if node.is_in_port_connected(in_port_with_initial_states):
            initial_hidden_state_size = node.in_port(in_port_with_initial_states).data.get_shape()
            if initial_hidden_state_size is not None:
                batch_size = initial_hidden_state_size[1]

    if seq_len is dynamic_dimension and node.format == 'onnx':
        # ONNX can store seq_len in optional input
        if node.is_in_port_connected(4):
            seq_len_val = node.in_port(4).data.get_value()
            if seq_len_val is not None:
                seq_len = seq_len.item()

    return [batch_size, seq_len]


def get_rnn_input_size(node: Node):
    node_name = node.soft_get('name', node.id)
    assert node.is_in_port_connected(1), 'weights input is not connected'

    if node.format == 'onnx':
        # ONNX weights on input 1 contain only W part, R, and B are connected separately
        # weights_shape = `[num_directions, 4 * hidden_size, input_size]`
        weights_size = node.in_port(1).data.get_shape()
        assert len(weights_size) == 3, 'incorrect weights ranks for ONNX {} node {}'.format(node.op, node_name)
        input_size = weights_size[2]
        return input_size
    elif node.format == 'tf':
        log.error('reverse infer for TensorFlow RNN operation {} is not implemented yet'.format(node_name),
                  extra={'is_warning': True})
    else:
        raise Error('Incorrect framework name')
