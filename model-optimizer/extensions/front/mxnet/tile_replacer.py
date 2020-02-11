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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const


class TileReplacer(FrontReplacementOp):
    op = 'Tile'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        if node.has_valid('reps'):
            tile_array = Const(graph, dict(value=int64_array(node.reps),
                                           symbol_dict={'name': node.id + '/tile_array'})).create_node()
            node.in_port(1).get_connection().set_source(tile_array.out_port(0))
