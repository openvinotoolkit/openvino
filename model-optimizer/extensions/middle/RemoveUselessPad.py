"""
 Copyright (C) 2020 Intel Corporation

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


class RemoveUselessPad(MiddleReplacementPattern):
    """
    The Pad layer is removed if all padding values are equal to 0 (Constant values).
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Pad'):
            all_pads_zeros = True
            for in_port_ind in range(1, 3):
                input_node = node.in_port(in_port_ind).get_source().node
                value = input_node.soft_get('value', None)
                all_pads_zeros &= input_node.soft_get('type') == 'Const' and value is not None and np.all(value == 0)

            if all_pads_zeros:
                remove_op_node_with_data_node(graph, node)
