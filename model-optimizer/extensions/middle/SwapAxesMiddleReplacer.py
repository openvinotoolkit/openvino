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
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class SwapAxesMiddleReplacer(MiddleReplacementPattern):
    enabled = False

    def pattern(self):
        return dict(
            nodes=[('swapaxes', dict(kind='op', op='swapaxes'))],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        """
            Replace swapaxes layer:
            swapaxes -> Reshape
        """

        swapaxes = match['swapaxes']
        swapaxes_in_node = swapaxes.in_node()
        swapaxes_out_node = swapaxes.out_node()

        input_edge_attrs = graph.get_edge_data(swapaxes_in_node.id, swapaxes.id)[0]
        output_edge_attrs = graph.get_edge_data(swapaxes.id, swapaxes_out_node.id)[0]

        graph.remove_edge(swapaxes_in_node.id, swapaxes.id)
        graph.remove_edge(swapaxes.id, swapaxes_out_node.id)
        Reshape(graph, {'dim': np.array(swapaxes_in_node.shape)}).create_node_with_data(inputs=[swapaxes_in_node],
                                                                                        data_nodes=[swapaxes_out_node],
                                                                                        edge_attrs=[input_edge_attrs,
                                                                                                    output_edge_attrs])
