# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.eltwise_n import EltwiseNReplacement
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const


class EltwiseAddNormalize(FrontReplacementPattern):
    """
    The Caffe layer "Eltwise" with operation SUM has optional attribute "coeff" which specifies the constant to multiply
    the inputs before applying. This transformation inserts Mul operation to the inputs and removes the "coeff"
    attribute from the node.
    """
    enabled = True

    def run_before(self):
        return [EltwiseNReplacement]

    @staticmethod
    def __insert_mul_node_with_coeff(node: Node, port: int, coeff: float):
        if coeff != 1:
            mul_node = Mul(node.graph, {'name': node.id + '/coeff_mul'}).create_node()
            const_node = Const(node.graph, {'name': node.id + '/coeff', 'value': mo_array([coeff])}).create_node()
            node.in_port(port).get_connection().insert_node(mul_node)
            const_node.out_port(0).connect(mul_node.in_port(1))

    def find_and_replace_pattern(self, graph: Graph):
        for eltwise_node in graph.get_op_nodes(op='EltwiseN', operation='sum') + graph.get_op_nodes(op='Add'):
            if eltwise_node.has_valid('coeff') and len(eltwise_node.coeff):
                coeff = eltwise_node.coeff

                for i in range(len(coeff)):
                    __class__.__insert_mul_node_with_coeff(eltwise_node, i, coeff[i])

                eltwise_node.coeff = None
                if len(coeff) > 2:
                    eltwise_node.op = "EltwiseN"
                    eltwise_node.type = "EltwiseN"
                    eltwise_node['operation'] = "sum"

