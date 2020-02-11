"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.ops.split import VariadicSplit
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.front.common.partial_infer.utils import int64_array
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape
from mo.ops.const import Const


class LSTMCellNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('lstm_cell', {'type': 'LSTMCell'})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['lstm_cell']
        lstm_cell_name = node.soft_get('name', node.id)
        hidden_size = node.get_attrs()["hidden_size"]

        WR_shape = node.in_port(3).data.get_shape()
        assert WR_shape is not None, "Undefined 'WR' input shape for LSTM Cell node '{}'".format(lstm_cell_name)

        num_elements_in_WR = np.prod(WR_shape)
        input_size = (num_elements_in_WR / (4 * hidden_size)) - hidden_size

        # Reshape
        reshape = create_op_node_with_second_input(graph, Reshape,
                                                   int64_array([4 * hidden_size, hidden_size + input_size]),
                                                   {'name': lstm_cell_name + '/Dims'})

        # VariadicSplit
        const_axis = Const(graph, {'value': 1}).create_node()
        const_size_splits = Const(graph, {'value': int64_array([input_size, hidden_size])}).create_node()
        split = VariadicSplit(graph, {'name': lstm_cell_name + '/Split', 'out_ports_count': 2}).create_node()
        const_axis.out_port(0).connect(split.in_port(1))
        const_size_splits.out_port(0).connect(split.in_port(2))

        # LSTM Cell
        node.in_port(3).get_connection().set_destination(reshape.in_port(0))
        reshape.out_port(0).connect(split.in_port(0))

        node.add_input_port(5, skip_if_exist=True)
        assert node.in_port(5).disconnected()
        node.in_port(4).get_connection().set_destination(node.in_port(5))

        split.out_port(0).connect(node.in_port(3))
        split.out_port(1).connect(node.in_port(4))
