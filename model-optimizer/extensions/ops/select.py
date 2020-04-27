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

from mo.front.common.partial_infer.utils import int64_array, broadcast_shape
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Select(Op):
    op = 'Select'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'type': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': __class__.infer,
            'type_infer': __class__.type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 3, "Select operation must have 3 inputs by TensorFlow reference:" \
                                          " \'condition\', \'then\' and \'else\' tensors"
        condition_node = node.in_node(0)
        resulting_tensors = [node.in_node(1), node.in_node(2)]

        a_shape = node.in_port(1).data.get_shape()
        b_shape = node.in_port(2).data.get_shape()
        output_shape = broadcast_shape(a_shape, b_shape)

        # Case with unknown condition
        if not condition_node.has_valid('value'):
            # infer only shapes
            node.out_port(0).data.set_shape(output_shape)
            return

        assert condition_node.value.size == 1
        condition_value = condition_node.value.item(0)

        assert isinstance(condition_value, np.bool), \
            "TensorFlow \'Select\' operation has 3 inputs: \'condition\', \'then\' and \'else\' tensors. " \
            "Value of \'condition\' tensor must be boolen by TensorFlow reference"

        output_value = resulting_tensors[not condition_value].value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = np.array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)

    @staticmethod
    def type_infer(node: Node):
        assert node.in_port(1).get_source().get_data_type() == node.in_port(2).get_source().get_data_type(), \
            'The data type of the second and the third inputs must be equal for the node {}'.format(node.name)
        node.out_port(0).set_data_type(node.in_port(1).get_source().get_data_type())
