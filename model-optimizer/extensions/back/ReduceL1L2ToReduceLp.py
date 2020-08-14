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
from extensions.back.ReduceMerge import ReduceMerge
from extensions.ops.ReduceOps import ReduceLp
from mo.back.replacement import BackReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_node


class ReduceL1L2ToReduceLp(BackReplacementPattern):
    """
    Replaces ReduceL1 and ReduceL2 operations with ReduceLp with third input equal to 1 or 2 respectively.
    """
    enabled = True

    def run_after(self):
        return [ReduceMerge]

    def find_and_replace_pattern(self, graph: Graph):
        for reduce in graph.get_op_nodes(op='ReduceL1') + graph.get_op_nodes(op='ReduceL2'):
            reduce_name = reduce.soft_get('name', reduce.id)
            p = 1 if reduce.op == 'ReduceL1' else 2
            reduce_lp = create_op_with_const_inputs(graph, ReduceLp, {2: p})
            reduce_lp.in_port(0).connect(reduce.in_port(0).get_source())
            reduce_lp.in_port(1).connect(reduce.in_port(1).get_source())
            reduce.out_port(0).get_connection().set_source(reduce_lp.out_port(0))

            graph.remove_node(reduce.id)
            rename_node(reduce_lp, reduce_name)
