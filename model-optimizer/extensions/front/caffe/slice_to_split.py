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

import numpy as np

from extensions.ops.splitv import SplitV
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph


class SliceToSplit(FrontReplacementOp):
    op = "Slice"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        name = node.soft_get('name', node.id)

        assert node.has_valid('axis'), 'Slice operation `{}` has no `axis` parameter'.format(name)
        axis = np.array(node.axis)
        if axis.size != 1:
            return

        assert node.has_valid('slice_point'), 'Slice operation `{}` has no `slice_point` parameter'.format(name)
        slice_point = np.array(node.slice_point)
        if slice_point.size == 0:
            return
        size_splits = []
        curr_pos = 0
        for point in slice_point:
            assert point > curr_pos
            size_splits.append(point - curr_pos)
            curr_pos = point
        size_splits.append(-1)

        split_node = SplitV(graph, {'name': name, 'size_splits': np.array(size_splits), 'axis': np.array(axis),
                                    'out_ports_count': len(slice_point) + 1}).create_node()

        node.in_port(0).get_connection().set_destination(split_node.in_port(0))
        for i, port in node.out_ports().items():
            node.out_port(i).get_connection().set_source(split_node.out_port(i))

        return []
