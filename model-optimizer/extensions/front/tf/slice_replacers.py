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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.eltwise import Eltwise
from mo.ops.slice import Slice
from extensions.ops.select import Select


class TFSliceToSliceReplacer(FrontReplacementOp):
    op = 'TFSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        This transformation converts TFSlice to internal Slice operation.
        In TFSlice node if size[i] is -1, all remaining elements in dimension i are included in the slice.
        In Slice which is borrowed from ONNX -1 is treated penultimate (shape[i] - 1). Also TFSlice has sizes
        on the second input while Slice has ends. This transformation was added to avoid multiple ifs in the future.
        """
        node = match['op']
        begin_node = node.in_node(1)
        size_node = node.in_node(2)

        eq_node = Eltwise(graph, dict(operation='equal', name=node.id + '/equal')).create_node()
        minus_one_node = Const(graph, dict(value=np.array(-1), name=node.id + '/minus_one')).create_node()
        int32_max_node = Const(graph, dict(value=np.iinfo(np.int32).max, name=node.id + '/int32_max')).create_node()
        select_node = Select(graph, dict(name=node.id + '/select')).create_node()

        # node to convert sizes to ends
        sum_node = Eltwise(graph, dict(operation='sum', name=node.id + '/end_const')).create_node()
        slice_node = Slice(graph, dict(name=node.id + '/slice_')).create_node()

        # reconnect input from tfslice to slice
        node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        # connect begin of tfslice to start of slice
        node.in_port(1).get_connection().set_destination(slice_node.in_port(1))

        # (size -> ends) connect begins and sizes to sum to evaluate ends for Slice
        begin_node.out_port(0).connect(sum_node.in_port(0))
        node.in_port(2).get_connection().set_destination(sum_node.in_port(1))

        # if size[i] == -1 when take int32_max as end[i]
        size_node.out_port(0).connect(eq_node.in_port(0))
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
