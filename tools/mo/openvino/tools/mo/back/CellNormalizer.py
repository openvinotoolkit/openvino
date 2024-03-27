# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.split import VariadicSplit
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, is_fully_defined
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape


class CellNormalizer(BackReplacementPattern):
    # This class splits WR input on W and R for LSTMCell, GRUCell, RNNCell

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('cell', dict(type=lambda type: type in ['LSTMCell', 'GRUCell', 'RNNCell']))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['cell']
        cell_name = node.soft_get('name', node.id)
        cell_type = node.soft_get('type')
        WR_input_id = node.soft_get('wr_input_id')
        hidden_size_coef = node.soft_get('gates_count')
        hidden_size = node.get_attrs()["hidden_size"]

        # default values for RNNCell/GRUCell
        additional_port_id = 4
        if cell_type == "LSTMCell":
            additional_port_id = 5

        WR_shape = node.in_port(WR_input_id).data.get_shape()
        assert WR_shape is not None, "Undefined 'WR' input shape for Cell node '{}'".format(cell_name)
        assert is_fully_defined(WR_shape), 'Not fully defined shape for WR for Cell node "{}"'.format(cell_name)

        num_elements_in_WR = np.prod(WR_shape)
        input_size = (num_elements_in_WR / (hidden_size_coef * hidden_size)) - hidden_size

        # Reshape
        reshape = create_op_node_with_second_input(graph, Reshape,
                                                   int64_array([hidden_size_coef * hidden_size,
                                                                hidden_size + input_size]),
                                                   {'name': cell_name + '/Dims'})

        # VariadicSplit
        split = create_op_with_const_inputs(graph, VariadicSplit, {1: int64_array(1),
                                                                   2: int64_array([input_size, hidden_size])},
                                            {'out_ports_count': 2, 'name': cell_name + '/Split'},
                                            reshape)

        # Cell
        node.in_port(WR_input_id).get_connection().set_destination(reshape.in_port(0))

        node.add_input_port(additional_port_id, skip_if_exist=True)
        assert node.in_port(additional_port_id).disconnected()

        # (x, y, WR, B) -> (x, y, W, R, B(additional_port))
        node.in_port(additional_port_id - 1).get_connection().set_destination(node.in_port(additional_port_id))
        split.out_port(0).connect(node.in_port(additional_port_id - 2))
        split.out_port(1).connect(node.in_port(additional_port_id - 1))
