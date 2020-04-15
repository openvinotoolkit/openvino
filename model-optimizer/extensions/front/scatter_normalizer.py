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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class ScatterNormalizer(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(is_scatter=True):
            name = node.soft_get('name', node.id)
            input_ports_count = len([port for port in node.in_ports().values() if not port.disconnected()])
            has_axis = node.has_valid('axis')

            if has_axis:
                assert input_ports_count == 3, \
                    '{} node {} has unexpected number of input ports {}'.format(node.op, name, input_ports_count)
                const = Const(graph, {'name': name + '/axis', 'value': np.int64(node.axis)}).create_node()
                node.add_input_port(3, skip_if_exist=True)
                node.in_port(3).connect(const.out_port(0))
                del node['axis']
            else:
                assert input_ports_count == 4, \
                    '{} node {} has unexpected number of input ports {}'.format(node.op, name, input_ports_count)
