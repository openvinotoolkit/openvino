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

import logging as log

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.utils.error import Error


class SqueezeNormalize(FrontReplacementPattern):
    """
    Normalizes inputs of the Squeeze layer.
    If axes is known the layer should have two inputs: the input with data and input with the axes to squeeze.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Squeeze'):
            if node.has_valid('axes') and not node.is_in_port_connected(1):
                name = node.soft_get('name', node.id)
                axes_node = Const(graph, {'name': name + '/axes', 'value': int64_array(node.axes)}).create_node()
                node.in_port(1).connect(axes_node.out_port(0))
                del node['axes']
