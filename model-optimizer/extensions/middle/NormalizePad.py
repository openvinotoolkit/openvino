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

from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class NormalizePad(MiddleReplacementPattern):
    """
    The replacer finds all Pad operations and remove inputs with index 1 and 2. These inputs contain padding values
    for each input tensor dimension and optionally the pad value for case of padding with a 'constant' mode.

    The Pad layer is removed if all padding values are equal to 0.
    """
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('pad', dict(kind='op', op='Pad'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pad']
        for port, input_node in node.in_nodes().items():
            if port != 0:
                graph.remove_edge(input_node.id, node.id)

        # remove Pad operation if all pads are equal to 0
        if np.all(node.pads == 0):
            remove_op_node_with_data_node(graph, node)
