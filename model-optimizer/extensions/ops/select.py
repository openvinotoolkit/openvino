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

from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error


class Select(Op):
    op = 'Select'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 3, "Select operation must have 3 inputs by TensorFlow reference:" \
                                          " \'condition\', \'then\' and \'else\' tensors"
        condition_node = node.in_node(0)
        resulting_tensors = [node.in_node(1), node.in_node(2)]

        assert np.array_equal(resulting_tensors[0].shape, resulting_tensors[1].shape), \
            "TensorFlow \'Select\' operation has 3 inputs: \'condition\', \'then\' and \'else\' tensors." \
            "\'then\' and \'else\' tensors must have the same shape by TensorFlow reference"
        output_shape = resulting_tensors[0].shape

        # Case with unknown condition
        if not condition_node.has_valid('value'):
            # infer only shapes
            for out in node.out_nodes():
                node.out_node(out).shape = np.array(output_shape)
            return

        condition_value = condition_node.value[0]

        assert isinstance(condition_value, np.bool_), \
            "TensorFlow \'Select\' operation has 3 inputs: \'condition\', \'then\' and \'else\' tensors. " \
            "Value of \'condition\' tensor must be boolen by TensorFlow reference"

        output_value = resulting_tensors[not condition_value].value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = np.array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)