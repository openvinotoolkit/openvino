# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins, shape_array, shape_insert
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node, add_opoutput, Graph
from openvino.tools.mo.ops.op import Op


class LSTMSequence(Op):
    """ Implements a layer that incorporates LSTM cell in a loop like it is specified in ONNX

        It is assumed that there is no equivalent of this op in IE,
        so it is considered as intermediate operation that will be translated differently.
        We define type for this operation to enable debuggin at OV side.

        There are several flavors of this op depending on how it was created and in which framework.
        There are several attributes that specifies the LSTM flavor:
            - ONNX/LSTM gives this op in non-normalized form and will require normalization
                as a separate transformation (see LSTMSequenceNormalize middle transformation);
                in this case blobs_wrb=True. Normalized weights/biases for MatMul is used when
                blobs_wrb=True.
            - ONNX/LSTM defines output shape as 4D: [seq_length, num_directions, batch_size,
                hidden_size], where num_directions = 1 is supported only. In this case
                has_num_directions=True. Otherwise, output is 3D and doesn't contain num_directions.
            - Depending on the original framework, `format` attrtibutes is specified accordingly.
                Its value controls which normalize transformations are called.
    """
    op = 'LSTMSequence'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,  # should be never emitted to IR; for debugging purposes
            'op': self.op,
            'blobs_wrb': False,
            'has_num_directions': False,
            'direction': 'forward',
            'num_layers': 1,
            'infer': self.infer,
            'blob_bidirectional_split': lambda node: (
                LSTMSequence.split_helper(node, 0, 'forward'),
                LSTMSequence.split_helper(node, 1, 'reverse')
            )
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'hidden_size',  # number of the elements in hidden cell size
            'direction',    # one of 'forward', 'reverse', or 'bidirectional'
            'batch_dim',    # batch dimension index in input/output shape
            'sequence_dim', # sequence dimension index in input shape
            'blobs_wrb',    # input blobs have three separate components W, R and B like in ONNX/LSTM
            'has_num_directions',  # if True, output shape has 4 dimensions; 3D otherwise
            'format',       # format type of input blobs for different frameworks (onnx, tf)
        ]

    def backend_attrs(self):
        return [
            'hidden_size',
            ('activations', lambda node: ','.join(node['activations']) if node.has_and_set('activations') else None),
            ('activations_alpha', lambda node: ','.join(map(str, node['activations_alpha']))
                if node.has_and_set('activations_alpha') else None),
            ('activations_beta', lambda node: ','.join(map(str, node['activations_beta']))
                if node.has_and_set('activations_beta') else None),
            'clip',
            'direction',
        ]

    @staticmethod
    def split_helper(node, index: int, direction: str):
        return Op._create_data_node(
            node.graph,
            name=node.name + '/SplittedBiLSTM/{}/'.format(direction),
            attrs={'value': node.value[index], 'shape': int64_array(node.value[index].shape)}
        )

    @staticmethod
    def infer(node: Node):
        # there are limitations coming from ONNX LSTM definition and normalization rules
        assert len(node.in_nodes()) >= 3  # X, W and R
        assert len(node.in_nodes()) <= 7
        assert len(node.out_nodes()) <= 3
        assert node.batch_dim <= 1
        assert node.sequence_dim <= 1
        assert node.batch_dim != node.sequence_dim

        assert node.direction in ['forward', 'reverse', 'bidirectional']

        if node.blobs_wrb:
            mark_input_bins(node, ['W', 'R', 'B'])
        else:
            mark_input_bins(node)
        input_shape = node.in_node(0).shape
        assert len(input_shape) == 3

        for port in [2, 3]:
            if port in node.in_nodes() and len(node.in_node(port).in_nodes()) > 0 and \
               'zero_shapes' in node.in_node(port).in_node():
                for i in node.in_node(port).in_node().zero_shapes:
                    if node.in_node(port).shape[i] != input_shape[i]:
                        node.in_node(port).value = np.repeat(node.in_node(port).value, input_shape[i], axis=i)
                        node.in_node(port).shape[i] = input_shape[i]

        out_shape = shape_array([input_shape[node.sequence_dim], input_shape[node.batch_dim], node.hidden_size])
        assert not node.has_num_directions or node.sequence_dim == 0, \
            'If has_num_directions == True, then node.sequence_dim should be equal 0, but it is {}'.format(
                node.sequence_dim)
        num_directions = 2 if node.direction in ['bidirectional'] else 1
        num_layers = node.num_layers
        if node.has_num_directions:
            # insert extra dimension to output shape for num_directions
            out_shape = shape_insert(out_shape, 1, np.int64(num_directions))
        node.out_node(0).shape = out_shape
        # extra outputs for hidden/cell states
        state_size = shape_array([input_shape[1], node.hidden_size])
        if node.has_num_directions:
            state_size = shape_insert(state_size, 0, num_directions * num_layers)
        for i in [1,2]:
            if i not in node.out_nodes():
                data_node = Op._create_data_node(
                    node.graph,
                    name=node.node+'/ExtraOutput/' + str(i),
                    attrs={'executable': True}
                )
                node.graph.add_edge(node.id, data_node.id, key=0, out=i)
                add_opoutput(node.graph, data_node.id, 0, False)
            else:
                data_node = node.out_node(i)
            data_node.shape = state_size.copy()
