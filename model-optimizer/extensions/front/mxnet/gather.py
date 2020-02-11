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
from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const


class GatherFrontReplacer(FrontReplacementOp):
    op = 'Embedding'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        gather_node = Gather(graph, dict(name=node.id + '/embedding_',
                                         symbol_dict={'name': node.id + '/embedding_'})).create_node()
        axis_const = Const(graph, {'value': int64_array(0)}).create_node()
        node.in_port(0).get_connection().set_destination(gather_node.in_port(1))
        node.in_port(1).get_connection().set_destination(gather_node.in_port(0))
        axis_const.out_port(0).connect(gather_node.in_port(2))
        node.out_port(0).get_connection().set_source(gather_node.out_port(0))
