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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph
from mo.utils.error import Error


class DisableUnsupportedNDOperations(BackReplacementPattern):
    """
        This pass disables ND Convolutions/Deconvolutions/Poolings
    """
    enabled = False

    unsupported_operations = ['Convolution', 'Deconvolution', 'Pooling']

    def find_and_replace_pattern(self, graph: Graph):
        unsupported_nodes = []
        for node in graph.nodes():
            node = Node(graph, node)
            if node.kind == 'op' and node.soft_get('type') in self.unsupported_operations:
                input_shape = node.in_node(0).shape
                if len(input_shape) > 4:
                    unsupported_nodes.append((node.id, node.type))

        if len(unsupported_nodes) == 0:
            return

        error_message = "\nOperations below were marked as unsupported due to they expect more than two spatial dims" \
                        " (input shape length more than 4)\n"
        error_message += "List of unsupported operations ({})\n".format(len(unsupported_nodes))
        for node, type in unsupported_nodes:
            error_message += "      {} {}\n".format(type, node)

        raise Error(error_message)
