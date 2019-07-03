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

import networkx as nx

from mo.front.common.partial_infer.utils import mark_input_bins
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
import numpy as np


class BlockLSTM(Op):
    op = 'BlockLSTM'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'infer': __class__.infer,
            'type': __class__.op,
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
         """
        assert len(node.in_nodes()) == 5

        """
        MO output edges:    |   Description:
                0           | cs: Output data / output hidden states concatenated over the whole time sequence
                1           | h: Output cell states concatenated over the whole time sequence
        """

        assert len(node.out_nodes()) in [1, 2]

        mark_input_bins(node)
        input_shape = node.in_node(0).shape

        assert len(input_shape) == 3
        out_shape = input_shape
        node.out_node(0).shape = out_shape
        if len(node.out_nodes()) > 1:
            node.out_node(1).shape = out_shape
