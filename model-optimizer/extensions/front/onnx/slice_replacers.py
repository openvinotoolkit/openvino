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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.slice import Slice


class AttributedSliceToSliceReplacer(FrontReplacementOp):
    """
    This class replaces AttributedSlice -> Slice
    """
    op = 'AttributedSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']

        slice_node = Slice(graph, {'name': node.id + '/slice_'}).create_node()
        node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        node.out_port(0).get_connection().set_source(slice_node.out_port(0))

        start_node = Const(graph, {'value': node.start, 'name': node.id + '/start_const'}).create_node()
        end_node = Const(graph, {'value': node.end, 'name': node.id + '/end_const'}).create_node()

        slice_node.in_port(1).get_connection().set_source(start_node.out_port(0))
        slice_node.in_port(2).get_connection().set_source(end_node.out_port(0))
        if node.has_valid('axis'):
            axis_node = Const(graph, {'value': node.axis, 'name': node.id + '/axis_const'}).create_node()
            # slice_node.add_input_port(3, skip_if_exist=True)
            slice_node.in_port(3).get_connection().set_source(axis_node.out_port(0))

