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

from extensions.ops.activation_ops import activation_ops
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.ops.activation import Activation


class ActivationsNormalizer(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('activation', dict(type=lambda type: type in activation_ops))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        node = match['activation']
        Activation.update_node_stat(node, dict(operation=node.type.lower()))
