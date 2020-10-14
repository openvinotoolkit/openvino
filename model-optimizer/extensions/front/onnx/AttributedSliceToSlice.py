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
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.slice import Slice


class AttributedSliceToSliceReplacer(FrontReplacementOp):
    """
    This class replaces AttributedSlice -> Slice
    """
    op = 'AttributedSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        slice_name = node.soft_get('name', node.id)

        slice_node = create_op_with_const_inputs(graph, Slice, {1: node.starts, 2: node.ends, 3: node.axes})
        rename_nodes([(node, slice_name + '/to_be_removed'), (slice_node, slice_name)])

        node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        node.out_port(0).get_connection().set_source(slice_node.out_port(0))
