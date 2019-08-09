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
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.eltwise import Eltwise
from mo.ops.eltwise_n import EltwiseN
from mo.ops.split import Split
from mo.utils.error import Error


class ReplaceEltwiseNin1NodePattern(FrontReplacementOp):
    """
    In nnet3 models Kaldi gather all inputs of Mul or Sum in 1. This pass separates inputs as it should be for IE.
    """
    op = "EltwiseNin1"
    enabled = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def replace_op(self, graph: Graph, node: Node):
        ss_node = Split(graph, attrs={'name': 'Split_eltwise_'+node.name,
                                      'num_split': node['num_inputs']}).create_node()

        inp = node.get_inputs()
        in_node = inp[0][0]
        edge_attrs = inp[0][1]
        graph.add_edge(in_node, ss_node.id, **edge_attrs)
        if ss_node.num_split == 2:
            eltwise_node = Eltwise(graph, attrs={'name': 'Eltwise_'+node.name,
                                                 'operation': node['operation']}).create_node()
        elif ss_node.num_split > 2:
            eltwise_node = EltwiseN(graph, attrs={'name': 'Eltwise_'+node.name,
                                                  'operation': node['operation']}).create_node()
        else:
            raise Error('Error on replacing Kaldi eltwise')
        for i in range(ss_node.num_split):
            ss_node.add_output_port(i)
            ss_node.out_port(i).get_connection().set_destination(eltwise_node.in_port(i))
        return [eltwise_node.id]
