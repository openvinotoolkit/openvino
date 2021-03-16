"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.front.common.partial_infer.utils import int64_array
from mo.ops.unsqueeze import Unsqueeze
from mo.ops.squeeze import Squeeze


class NonConstBeginStridedSliceReplacement(FrontReplacementSubgraph):
    """
    """
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[#('delta', dict(op='Identity')),
                   ('y', dict(op='Const')),
                   ('add', dict(op='Add')),
                   ('begin_0', dict(op='Const')),
                   ('begin_2', dict(op='Const')),
                   ('begin', dict(op='Pack')),
                   ('end_0', dict(op='Const')),
                   ('end_2', dict(op='Const')),
                   ('end', dict(op='Pack')),
                   ('step', dict(op='Const')),
                   ('strided_slice', dict(op='StridedSlice')),
                   ],
            edges=[#('delta', 'add'),
                   ('y', 'add', {'in': 1}),
                   ('begin_0', 'begin', {'in': 0}),
                   #('delta', 'begin', {'out': 0, 'in': 1}),
                   ('begin_2', 'begin', {'in': 2}),
                   ('end_0', 'end', {'in': 0}),
                   ('add', 'end', {'in': 1}),
                   ('end_2', 'end', {'in': 2}),
                   ('begin', 'strided_slice', {'in': 1}),
                   ('end', 'strided_slice', {'in': 2}),
                   ('step', 'strided_slice', {'in': 3})]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        add_node = match['add']
        strided_slice_node = match['strided_slice']
        strided_slice_name = strided_slice_node.soft_get('name', strided_slice_node.id)

        # unsqueeze a scalar by which to slice input tensor
        unsqueeze_axis = Const(graph, {'name': strided_slice_name + '/Unsqueeze/axis', 'value': int64_array(0)}).create_node()
        unsqueeze_node = Unsqueeze(graph, {'name': strided_slice_name + '/Unsqueeze'}).create_node()
        add_node.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
        unsqueeze_node.in_port(1).connect(unsqueeze_axis.out_port(0))

        # replace StridedSlice with Gather operation
        axis = Const(graph, {'name': strided_slice_name + '/axis', 'value': int64_array(1)}).create_node()
        gather_node = Gather(graph, {'name': strided_slice_name + '/Gather'}).create_node()
        axis.out_port(0).connect(gather_node.in_port(2))

        strided_slice_node.in_port(0).get_connection().set_destination(gather_node.in_port(0))
        unsqueeze_node.out_port(0).connect(gather_node.in_port(1))

        # squeeze Gather's output
        squeeze_node = Squeeze(graph, {'name': strided_slice_name}).create_node()
        squeeze_axis = Const(graph, {'name': strided_slice_name + '/Squeeze/axis', 'value': int64_array(1)}).create_node()
        squeeze_node.in_port(0).connect(gather_node.out_port(0))
        squeeze_node.in_port(1).connect(squeeze_axis.out_port(0))
        rename_nodes(
            [(strided_slice_node, strided_slice_name + '/AbandonedName'), (squeeze_node, strided_slice_name)])

        strided_slice_node.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
        graph.remove_nodes_from([match['add'].id, match['strided_slice'].id])
