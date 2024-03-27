# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins, dynamic_dimension, shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class BlockLSTM(Op):
    op = 'BlockLSTM'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'infer': self.infer,
            'type': None,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        """
         MO input edges:   |   Description:
         -------------------------------------------------
                0          | x: The sequence input to the LSTM, shape (timelen, batch_size, num_inputs)
                1          | w: The weight matrix
                2          | b: The bias vector
                3          | h_prev: Previous/initial hidden state
                4          | cs_prev: Value of the initial cell state

         MO output edges:    |   Description:
                0           | hs: Output data / output hidden states concatenated over the whole time sequence
                1           | cs: Output cell states concatenated over the whole time sequence
        """

        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port_idx for port_idx, port in node.in_ports().items() if not port.disconnected()]
        connected_out_ports = [port_idx for port_idx, port in node.out_ports().items() if not port.disconnected()]
        assert len(connected_in_ports) >= 5, "Internal Model Optimizer Error or unsupported BlockLSTM node {}. " \
                                             "MO expects five inputs for BlockLSTM".format(node_name)
        assert len(connected_out_ports) <= 2, "Internal Model Optimizer Error or unsupported BlockLSTM node {}. " \
                                              "MO expects at most two outputs for BlockLSTM".format(node_name)

        x_shape = node.in_port(0).data.get_shape()
        w_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()

        time_len = dynamic_dimension
        batch_size = dynamic_dimension
        if len(x_shape) > 2:
            time_len = x_shape[0]
            batch_size = x_shape[1]

        hidden_size_output = dynamic_dimension
        if len(b_shape) > 0 and b_shape[0] is not dynamic_dimension:
            hidden_size_output = b_shape[0] // 4
        elif len(w_shape) > 1 and w_shape[1] is not dynamic_dimension:
            hidden_size_output = w_shape[1] // 4

        # mark-up inputs for LSTMRNNSequenceToTensorIterator transformation
        mark_input_bins(node)

        x_output_shape = shape_array([time_len, batch_size, hidden_size_output])
        if node.is_out_port_connected(0):
            node.out_port(0).data.set_shape(x_output_shape)

        # at this point cell states are in aggregated form from all time steps
        # after that the middle transformation BlockLSTMtoLSTMSequence should normalize it to last step cell state
        if node.is_out_port_connected(1):
            node.out_port(1).data.set_shape(x_output_shape)
