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

from extensions.ops.elementwise import Add, Equal
from extensions.ops.select import Select
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.slice import Slice


class TFSliceToSliceReplacer(FrontReplacementOp):
    """
    This transformation converts TFSlice to internal Slice operation.
    In TFSlice size[i] == -1 means take all elements on axis i up to the end including(!) the last
    In internal MO Slice (which is borrowed from ONNX) -1 means take all excluding(!) the last (shape[i] - 1).
    Also TFSlice has 'sizes' on the second input while Slice has 'ends'.
    This transformation was added to avoid multiple if statements in future transformations.
    """
    op = 'TFSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        slice_name = node.soft_get('name', node.id)
        slice_node = Slice(graph).create_node()
        rename_nodes([(node, slice_name + '/to_be_removed'), (slice_node, slice_name)])

        eq_node = Equal(graph, {'name': slice_name + '/equal'}).create_node()
        minus_one_node = Const(graph, {'name': slice_name + '/minus_one', 'value': np.array(-1)}).create_node()
        int32_max_node = Const(graph, {'name': slice_name + '/int32_max', 'value': np.iinfo(np.int32).max}).create_node()
        select_node = Select(graph, {'name': slice_name + '/select'}).create_node()

        # node to convert sizes to ends
        sum_node = Add(graph, {'name': slice_name + '/end_const'}).create_node()

        # reconnect input from tfslice to slice
        node.in_port(0).get_source().connect(slice_node.in_port(0))
        node.in_port(0).disconnect()
        # reconnect begin of tfslice to start of slice
        node.in_port(1).get_source().connect(slice_node.in_port(1))
        node.in_port(1).disconnect()

        # (size -> ends) reconnect begins and sizes to sum to evaluate ends for Slice
        # connects begins to slice
        slice_node.in_port(1).get_source().connect(sum_node.in_port(0))
        node.in_port(2).get_source().connect(sum_node.in_port(1))
        node.in_port(2).disconnect()

        # if size[i] == -1 when take int32_max as end[i]
        sum_node.in_port(1).get_source().connect(eq_node.in_port(0))
        minus_one_node.out_port(0).connect(eq_node.in_port(1))
        # from equal to 0 port of select
        eq_node.out_port(0).connect(select_node.in_port(0))
        # from int32_max to 1 of select
        int32_max_node.out_port(0).connect(select_node.in_port(1))
        # from sum to 2nd of select
        sum_node.out_port(0).connect(select_node.in_port(2))
        # out of select to end (2nd of slice)
        select_node.out_port(0).connect(slice_node.in_port(2))

        node.out_port(0).get_connection().set_source(slice_node.out_port(0))
