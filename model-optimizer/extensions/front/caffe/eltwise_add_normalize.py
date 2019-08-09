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

from extensions.front.eltwise_n import EltwiseNReplacement
from extensions.ops.elementwise import Mul
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class EltwiseAddNormalize(FrontReplacementOp):
    """
    The Caffe layer "Eltwise" with operation SUM has optional attribute "coeff" which specifies the constant to multiply
    the inputs before applying. This transformation inserts Mul operation to the inputs and removes the "coeff"
    attribute from the node.
    """
    op = 'Add'
    enabled = True

    def run_before(self):
        return [EltwiseNReplacement]

    @staticmethod
    def __insert_mul_node_with_coeff(node: Node, port: int, coeff: float):
        if coeff != 1:
            mul_node = Mul(node.graph, {'name': node.id + '/coeff_mul'}).create_node()
            const_node = Const(node.graph, {'name': node.id + '/coeff', 'value': np.array([coeff])}).create_node()
            node.in_port(port).get_connection().insert_node(mul_node)
            const_node.out_port(0).connect(mul_node.in_port(1))

    def replace_sub_graph(self, graph: Graph, match: dict):
        eltwise_node = match['op']
        if eltwise_node.has_valid('coeff') and len(eltwise_node.coeff):
            coeff = eltwise_node.coeff

            for i in range(len(coeff)):
                __class__.__insert_mul_node_with_coeff(eltwise_node, i, coeff[i])

            eltwise_node.coeff = None

            if len(coeff) > 2:
                eltwise_node.op = "EltwiseN"
                eltwise_node.type = "EltwiseN"
                eltwise_node['operation'] = "sum"
