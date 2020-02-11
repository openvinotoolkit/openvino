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

from extensions.ops.mxrepeat import MXRepeat
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const


class ArangeReplacer(FrontReplacementOp):
    op = 'Range'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        if not node.has_valid('start') or not node.has_valid('stop') or not node.has_valid('step'):
            return

        start_value = Const(graph, dict(value=node.start,
                                         symbol_dict={'name': node.id + '/const_start'})).create_node()
        limit_value = Const(graph, dict(value=node.stop,
                                        symbol_dict={'name': node.id + '/const_limit'})).create_node()
        delta_value = Const(graph, dict(value=node.step,
                                        symbol_dict={'name': node.id + '/const_delta'})).create_node()
        node.in_port(0).get_connection().set_source(start_value.out_port(0))
        node.in_port(1).get_connection().set_source(limit_value.out_port(0))
        node.in_port(2).get_connection().set_source(delta_value.out_port(0))
        if node.has_valid('repeat') and node.repeat > 1:
            rep = MXRepeat(graph, dict(name=node.id + '/mxrepeat', axis=0, repeats=node.repeat)).create_node()
            node.out_port(0).get_destination().get_connection().set_source(rep.out_port(0))
            rep.in_port(0).connect(node.out_port(0))
