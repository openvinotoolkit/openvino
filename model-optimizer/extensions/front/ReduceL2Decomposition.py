"""
 Copyright (C) 2020 Intel Corporation

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
from extensions.front.reduce_axis_normalizer import ReduceAxisNormalizer
from extensions.ops.ReduceOps import ReduceSum
from extensions.ops.elementwise import Pow, Mul
from mo.front.common.partial_infer.utils import int64_array, float_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_node


class ReduceL2Decomposition(FrontReplacementOp):
    op = 'ReduceL2'
    enabled = True

    def run_before(self):
        return [ReduceAxisNormalizer]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)

        rename_node(node, node_name + '/TBR')
        sqr_node = Mul(graph, {}).create_node()
        reduce_sum_node = ReduceSum(graph, {'keep_dims': node.soft_get('keep_dims', 0),
                                            'axis': node.soft_get('axis', None)}).create_node()
        sqrt_node = create_op_with_const_inputs(graph, Pow, {1: float_array(0.5)})
        rename_node(sqrt_node, node_name)

        # Connect nodes
        node.in_port(0).get_connection().set_destination(sqr_node.in_port(0))
        sqr_node.in_port(0).get_connection().add_destination(sqr_node.in_port(1))
        sqr_node.out_port(0).connect(reduce_sum_node.in_port(0))
        reduce_sum_node.out_port(0).connect(sqrt_node.in_port(0))

        return [sqrt_node.id]
