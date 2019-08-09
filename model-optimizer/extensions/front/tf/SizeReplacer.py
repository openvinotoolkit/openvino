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
from extensions.ops.ReduceOps import ReduceProd
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.shape import Shape


class SizeFrontReplacer(FrontReplacementOp):
    """
    Replace Size op by Shape -> ReduceProd operations
    """
    op = "Size"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        shape = Shape(graph, {'name': node.name + '/Shape/'}).create_node()
        reduce_prod = ReduceProd(graph, {'name': shape.name + 'ReduceProd/', 'keep_dims': False}).create_node()
        reduce_axis = Const(graph, {'value': int64_array([0])}).create_node()

        # Connect nodes
        node.in_port(0).get_connection().set_destination(shape.in_port(0))
        reduce_prod.in_port(0).get_connection().set_source(shape.out_port(0))
        reduce_prod.in_port(1).get_connection().set_source(reduce_axis.out_port(0))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [reduce_prod.id]
