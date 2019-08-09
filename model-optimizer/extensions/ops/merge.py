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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Merge(Op):
    op = 'Merge'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'infer': __class__.merge_infer,
            'cf_infer': __class__.control_flow_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def merge_infer(node: Node):
        # we infer only through executable input nodes
        inferred_nodes = [n for n in node.in_nodes().values() if n['is_partial_inferred']]
        assert len(inferred_nodes) != 0

        if len(inferred_nodes) < len(node.in_nodes()):
            node['is_not_fully_inferred'] = True
        else:
            node['is_not_fully_inferred'] = False
            assert np.all(node.shape == inferred_nodes[0].shape for node in inferred_nodes)

            inferred_and_executable = [n for n in node.in_nodes().values() if n['is_partial_inferred'] and
                                       'executable' in n and n['executable']]
            tensor = inferred_and_executable[0]

            if all([np.all(tensor.value == n.value) for n in inferred_and_executable]):
                node.out_node().value = tensor.value.copy() if tensor.has_valid('value') else None

        tensor = inferred_nodes[0]
        node.out_node().shape = int64_array(tensor.shape)

    @staticmethod
    def control_flow_infer(node: Node, is_executable: bool, mark_executability: callable):
        graph = node.graph

        in_data_nodes = node.in_nodes(control_flow=True)
        out_data_nodes = node.out_nodes(control_flow=True)

        is_executable = any([d.has_and_set('executable') for i, d in in_data_nodes.items()]
                            if len(in_data_nodes) else [False])

        for i, d in out_data_nodes.items():
            mark_executability(d.id, is_executable)

