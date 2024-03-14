# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.eltwise_n import EltwiseN
from openvino.tools.mo.utils.error import Error


class ReplaceEltwiseNin1NodePattern(FrontReplacementOp):
    """
    In nnet3 models Kaldi gather all inputs of Mul or Sum in 1. This pass separates inputs as it should be for IE.
    """
    op = "EltwiseNin1"
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.restore_ports import RestorePorts
        return [RestorePorts]

    def replace_op(self, graph: Graph, node: Node):
        ss_node = create_op_with_const_inputs(graph, Split, {1: int64_array(1)}, {'name': 'Split_eltwise_' + node.name,
                                                                                  'num_splits': node['num_inputs']})

        inp = node.get_inputs()
        in_node = inp[0][0]
        edge_attrs = inp[0][1]
        graph.add_edge(in_node, ss_node.id, **edge_attrs)
        if ss_node.num_splits == 2:
            if node['operation'] == 'mul':
                eltwise_node = Mul(graph, attrs={'name': 'Eltwise_' + node.name}).create_node()
            elif node['operation'] == 'sum':
                eltwise_node = Add(graph, attrs={'name': 'Eltwise_' + node.name}).create_node()
            else:
                raise Error('Error on replacing Kaldi eltwise: unknown type ' + node['operation'])
        elif ss_node.num_splits > 2:
            eltwise_node = EltwiseN(graph, attrs={'name': 'Eltwise_' + node.name,
                                                  'operation': node['operation']}).create_node()
        else:
            raise Error('Error on replacing Kaldi eltwise')
        for i in range(ss_node.num_splits):
            ss_node.out_port(i).get_connection().set_destination(eltwise_node.in_port(i))
        return [eltwise_node.id]
