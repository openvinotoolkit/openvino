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


class Merge(Op):
    op = 'Merge'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.merge_infer
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
        node.out_node().shape = tensor.shape
