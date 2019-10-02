"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.pass_separator import FrontStart
from extensions.front.restore_ports import RestorePorts
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const


class MoveEmbeddedInputsToInputs(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        return [RestorePorts]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(kind='op', embedded_inputs=lambda x: x is not None))],
            edges=[]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        node = match['op']
        for port_index, value_attr, attrs in node['embedded_inputs']:
            const = Const(graph, dict(value=node[value_attr])).create_node()
            node.add_input_port(port_index, skip_if_exist=True)
            const.out_port(0).connect(node.in_port(port_index))
            node.in_port(port_index).bin = attrs['bin']
            node.in_port(port_index).in_attrs.append('bin')
            del node[value_attr]
        del node['embedded_inputs']
