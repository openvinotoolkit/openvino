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

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Add, Equal
from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.slice import Slice
from mo.front.tf.graph_utils import create_op_with_const_inputs


class TFSliceToSliceReplacer(FrontReplacementOp):
    """
    This transformation converts TFSlice to internal Slice operation.
    TFSlice has 'size' on the second input while Slice has 'ends' -> need Add(begin, size)
    size[i] == -1 is a magic number that means take the whole range along axis i up to the end
    therefore need to insert subgraph with ShapeOf to process that case
    """
    op = 'TFSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        tf_slice_node = match['op']
        slice_name = tf_slice_node.soft_get('name', tf_slice_node.id)
        slice_node = Slice(graph).create_node()
        rename_nodes([(tf_slice_node, slice_name + '/to_be_removed'), (slice_node, slice_name)])
        add_node = Add(graph, {'name': slice_name + '/end_const'}).create_node()

        # reconnect begin and input from TFSlice to Slice
        tf_slice_node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        tf_slice_node.in_port(1).get_connection().set_destination(slice_node.in_port(1))
        tf_slice_node.in_port(2).get_connection().set_destination(add_node.in_port(0))
        slice_node.in_port(1).get_connection().add_destination(add_node.in_port(1))

        shapeof_node = Shape(graph, {'name': slice_name + '/ShapeOf'}).create_node()
        slice_node.in_port(0).get_connection().add_destination(shapeof_node.in_port(0))

        # nodes to check if size == -1, if so take the whole range
        eq_node = create_op_with_const_inputs(graph, Equal, {0: int64_array(-1)}, {'name': slice_name + '/equal'})
        add_node.in_port(0).get_connection().add_destination(eq_node.in_port(1))

        # select requires equal dtypes
        cast_node = Cast(graph, {'name': slice_name + '/CastToI64', 'dst_type': np.int64}).create_node([add_node])
        select_node = Select(graph, {'name': slice_name + '/select'}).create_node([eq_node, shapeof_node, cast_node])
        select_node.out_port(0).connect(slice_node.in_port(2))

        tf_slice_node.out_port(0).get_connection().set_source(slice_node.out_port(0))
