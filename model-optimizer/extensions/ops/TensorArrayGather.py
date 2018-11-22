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
from mo.utils.utils import match_shapes


class TensorArrayGather(Op):
    op = "TensorArrayGatherV3"

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': TensorArrayGather.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        assert len(node.in_nodes()) == 3

        handle = node.in_node(0)
        indices = node.in_node(1)
        flow_in = node.in_node(2)

        ta_node = Node(node.graph, str(handle.value))

        if ta_node.has_valid('element_shape'):
            assert match_shapes(ta_node['element_shape'], node.element_shape)
        ta_node['element_shape'] = node.element_shape
        data_shape = ta_node['element_shape']
        if -1 in data_shape:
            assert data_shape.size == 2 and data_shape[0] == -1 and data_shape[1] != -1
            # This is a workaround for a bug that we cannot deduce element_shape
            # when it is required for inference, so we putting 1 here instead of -1.
            # This makes impossible to have batch size > 1 for LSTM-like loops
            data_shape[0] = 1
            ta_node.element_shape = data_shape

        assert ta_node.has_valid('size')
        size = ta_node['size']

        output_shape = [size] + [data_shape[i] for i in range(len(data_shape))]
        output_value = None

        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = np.array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)
