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

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class TensorArray(Op):
    op = "TensorArrayV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': TensorArray.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        size = node.in_node(0)
        assert size.value is not None

        # 0 port: handle
        if 0 in node.out_nodes().keys():
            if node.has_valid('element_shape'):
                element_shape = node['element_shape']
            else:
                element_shape = None

            out_node = node.out_node(0).id
            output_value = node.out_node(0).id
            node.graph.node[out_node]['value'] = np.array(output_value)

            output_shape = node.graph.node[out_node]['value'].shape
            node.graph.node[out_node]['shape'] = np.array(output_shape)

            node.graph.node[out_node]['element_shape'] = np.array(element_shape)
            node.graph.node[out_node]['size'] = size.value
        # 1 port flow
        if 1 in node.out_nodes().keys():
            output_value = None

            out_node = node.out_node(1).id
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)
            node.graph.node[out_node]['shape'] = np.array(output_shape)
